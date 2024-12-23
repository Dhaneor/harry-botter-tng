#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2024 17:43:33 2024

@author dhaneor
"""
import asyncio
import datetime
import logging
from typing import Any
from .base import BaseTable, DatabaseManager
from src.data.rawi import ohlcv_repository as repo
from src.util.timeops import interval_to_milliseconds, seconds_to

logger = logging.getLogger(f"main.{__name__}")

BATCH_SIZE = 10_000  # Batch size for bulk insert


class OhlcvTable(BaseTable):
    def __init__(
        self,
        db_manager: DatabaseManager,
        exchange: str,
        symbol: str,
        interval: str
    ):
        super().__init__(db_manager)
        self.exchange = exchange
        self.symbol = "".join(symbol.split("/"))
        self.interval = interval
        self.table_name = f"{self.exchange}_{self.symbol}_{self.interval}"
        logger.info(f"Creating OHLCV table: {self.table_name} for {exchange} {symbol}")

    async def create(self):
        """Creates the OHLCV table if it doesn't exist.

        The data returned by the Binance API looks like this:
        [
        [
            1499040000000,      // Kline open time
            "0.01634790",       // Open price
            "0.80000000",       // High price
            "0.01575800",       // Low price
            "0.01577100",       // Close price
            "148976.11427815",  // Volume
            1499644799999,      // Kline Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "0"                 // Unused field, ignore.
        ]
        ]

        We can get rid of the last field (unused) for our table.
        """

        query = f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            openTime BIGINT NOT NULL UNIQUE,
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(30, 8) NOT NULL,
            closeTime BIGINT NOT NULL,
            quoteAssetVolume DECIMAL(30, 8) NOT NULL,
            numberOfTrades INT,
            takerBuyBaseAssetVolume DECIMAL(30, 8),
            takerBuyQuoteAssetVolume DECIMAL(30, 8)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        await self.db.execute(query)

    async def insert(self, data, retries=3):
        """
        Insert one or multiple rows into the table.

        Arguments:
        ----------
        data
            - A list of values for a single row (List[Any])
            - A list of rows, where each row is a list of values (List[List[Any]])
        """
        if not data:
            logger.warning("No data to insert.")
            return  # No data to insert

        await self._fetch_columns()  # Ensure columns are fetched and cached
        logger.debug(f"Columns: {self.columns}")

        # Convert the list of lists to a list of dictionaries
        dict_data = []
        for row in data:
            logger.debug(row)
            dict_row = {
                col: val for col, val in zip(self.columns[1:], row)
                }  # Skip 'id' column
            dict_data.append(dict_row)

        columns = [col for col in self.columns if col.lower() != 'id']

        for attempt in range(retries):
            try:
                await super().insert(dict_data, columns)
                logger.debug(
                    "INSERT OK [%s row] for %s",
                    len(data) if isinstance(data[0], list) else 1,
                    self.table_name
                    )
                return
            except ValueError as ve:
                logger.error(
                    "[%s] INSERT FAIL for %s -> %s ",
                    attempt, self.table_name, str(ve)
                    )
                raise
            except Exception as e:
                logger.error(
                    "[%s] INSERT FAIL for %s -> %s ",
                    attempt, self.table_name, str(e)
                    )
                await asyncio.sleep(2)
                if attempt == retries - 1:  # Last retry
                    logger.error("🚨 All retries failed for insert.")
                    raise

    async def fetch_latest(self, limit: int = 1000) -> list[list[Any]]:
        query = f"""
        SELECT * FROM {self.table_name}
        ORDER BY openTime ASC
        LIMIT :limit
        """
        result = await self.db.fetch_all(query, values={"limit": limit})
        return self._to_list_of_lists(result)

    async def fetch_by_range(self, start: int, end: int) -> list[list[Any]]:
        query = f"""
        SELECT * FROM {self.table_name}
        WHERE openTime BETWEEN :start AND :end
        ORDER BY openTime ASC
        """
        result = await self.db.fetch_all(query, {"start": start, "end": end})
        return self._to_list_of_lists(result)

    async def get_row_count(self) -> int:
        """
        Returns the number of rows in the table.
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        result = await self.db.fetch_one(query)
        return result[0] if result else 0

    async def get_last_entry_ts(self) -> int:
        """
        Returns the timestamp of the last entry in the table.
        """
        query = f"SELECT MAX(openTime) FROM {self.table_name}"
        result = await self.db.fetch_one(query)
        return result[0] if result else 0

    async def needs_update(self) -> bool:
        """
        Checks if the table is up to date based on the latest entry
        and the interval for this table.

        Returns
        -------
        bool
            True if the table is not up to date, False otherwise
        """
        interval_ms = interval_to_milliseconds(self.interval)
        latest_open_ms = await self.get_last_entry_ts()
        latest_close_ms = latest_open_ms + interval_ms
        now_ms = datetime.datetime.now().timestamp() * 1000
        delta = (now_ms - latest_close_ms)

        logger.debug(
            "now: %s, latest close: %s, time delta: %s interval: %s",
            latest_close_ms, now_ms, f"{int(delta):,}", f"{interval_ms:,}",
            )

        latest_close_utc = (
            datetime
            .datetime
            .fromtimestamp(latest_close_ms / 1000, datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        now_utc = (
            datetime
            .datetime
            .now(datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        needs_update = delta > interval_ms

        logger.debug(
            "now_utc: %s, latest_close_utc: %s, "
            "time delta: %s, interval: %s, needs update: %s",
            now_utc, latest_close_utc,
            seconds_to(delta / 1000), seconds_to(interval_ms / 1000), needs_update
            )

        return needs_update

    async def update(self, end: int = None) -> None:
        """
        Updates the table with new data from the Binance API.

        Arguments:
        ----------
        end
            - The timestamp of the last entry to be included in the update.
            - If not provided, the update will include all available data.
        """
        logger.info(f"Updating OHLCV table: {self.table_name}")

        if end is None:
            end = datetime.datetime.now().timestamp() * 1000

        start = await self.get_last_entry_ts() + 1

        logger.info(
            "updating from %s to %s",
            seconds_to(start / 1000), seconds_to(end / 1000)
            )

        request = {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "start": int(start),
            "end": int(end),
        }

        response = await repo.process_request(request)

        if not response:
            logger.error(f"No data received for {self.table_name}")
            return

        if not response.success:
            logger.error(f"Error fetching data from {self.table_name}: {response.error}")
            return

        logger.info(f"Received {len(response)} rows for {self.table_name}")
        await self.insert(response.data)