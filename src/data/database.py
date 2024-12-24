#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database management module for the Harry Botter Trading System.

This module provides base classes and utilities for database operations,
including connection management and table operations. It uses the `databases`
library for asynchronous database access.

Environment Variables:
    The following environment variables must be set for database connection:
    - DB_USER: Database username
    - DB_PASS: Database password
    - DB_HOST: Database host (default: "localhost")
    - DB_PORT: Database port (default: 3306)
    - DB_NAME: Database name (default: "akasha")

    These can be set in a .env file or directly in the environment.

Classes:
    - DatabaseManager: Manages database connections and provides access to the database.
    - BaseTable: Abstract base class for database tables, providing common operations.

Note:
    Ensure that all required environment variables are set before using this module.
    The module uses dotenv to load environment variables from a .env file if present.

Usage:
    from data.database.base import DatabaseManager, BaseTable

    # Create a database manager
    db_manager = DatabaseManager()

    # Create a table class inheriting from BaseTable
    class MyTable(BaseTable):
        ...

    # Use the table class with the database manager
    my_table = MyTable(db_manager, exchange, symbol, interval)

Created on Fri Dec 202 15:50:33 2024

@author dhaneor
"""
import asyncio
import datetime
import logging
import os
import pymysql
import warnings
from abc import abstractmethod
from databases import Database
from dotenv import load_dotenv
from typing import Any, Coroutine
from urllib.parse import quote_plus

from .data_models import Ohlcv
from .ohlcv_repository import process_request
from util.timeops import interval_to_milliseconds, seconds_to

# Ignore pymysql warning about duplicate entries in the same table
# This is handled by the appropriate SQL INSERT IGNORE statement.
warnings.filterwarnings('ignore', category=pymysql.Warning,
                        message=r".*Duplicate entry.*for key.*")

logger = logging.getLogger(f"main.{__name__}")

# Load environment variables from .env file
load_dotenv()

logger.debug(f"DB_USER: {os.getenv("DB_USER")}")
logger.debug(f"DB_PASSWORD: {os.getenv("DB_PASS")}")
logger.debug(f"DB_HOST: {os.getenv("DB_HOST")}")
logger.debug(f"DB_PORT: {os.getenv("DB_PORT")}")
logger.debug(f"DB_NAME: {os.getenv("DB_NAME", "akasha")}")


class BaseTable:
    table_name: str = ""

    def __init__(self, db_manager: "DatabaseManager", repo: Coroutine):
        self.db = db_manager.db
        self.repo = repo
        self.columns: list[str] = []

    async def exists(self) -> bool:
        query = f"SHOW TABLES LIKE '{self.table_name}'"
        result = await self.db.fetch_one(query)
        return bool(result)

    @abstractmethod
    async def create(self):
        ...

    async def drop(self) -> None:
        query = f"DROP TABLE IF EXISTS {self.table_name}"
        await self.db.execute(query)

    async def insert(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        columns: list[str]
    ) -> None:
        # prepare the columns and placeholders for the SQL query
        columns_formatted = ', '.join(columns)
        placeholders = ', '.join([f':{col}' for col in columns])

        # build the SQL query
        query = (
            f"INSERT IGNORE INTO {self.table_name} "
            f"({columns_formatted}) VALUES ({placeholders})"
        )

        # Ensure each dictionary in data doesn't include the 'id' key
        cleaned_data = [
            {k: v for k, v in row.items() if k.lower() != 'id'} for row in data
            ]

        # Check if data is a single row or multiple rows
        if isinstance(data[0], dict):
            logger.debug(f"BATCH INSERT for {self.table_name} with {len(data)} entries")
            logger.debug(placeholders)
            logger.debug(data[0])
            # Batch insert
            if len(data[0]) != len(self.columns) - 1:
                raise ValueError(
                    f"Length of data {len(cleaned_data[0])} and number of "
                    f"columns {len(placeholders)} does not match."
                    )
            else:
                logger.debug(
                    "no of columns: %s, items in row: %s",
                    len(placeholders.split(' ')), len(cleaned_data[0])
                    )
            await self.db.execute_many(query, cleaned_data)
        else:
            # Single row insert
            if len(data) != len(self.columns) - 1:
                raise ValueError(
                    f"Length of data {len(data)} and number of "
                    f"columns {len(self.columns) - 1} does not match."
                    )

            await self.db.execute(query, data)

    async def fetch_all(self) -> list[list[Any]]:
        query = f"SELECT * FROM {self.table_name}"
        return self._to_list_of_lists(
            await self.db.fetch_all(query)
        )

    async def fetch_by(self, **conditions) -> list[dict[str, Any]]:
        where_clause = " AND ".join([f"{key} = :{key}" for key in conditions.keys()])
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        return self._to_list_of_lists(
            await self.db.fetch_all(query, conditions)
        )

    # ................................................................................
    async def _fetch_columns(self):
        """
        Fetch column names from the database schema and cache them.
        """
        if self.columns:
            return self.columns  # Return cached columns

        query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{self.table_name}'
        """
        result = await self.db.fetch_all(query)
        self.columns = [row['COLUMN_NAME'] for row in result]
        if not self.columns:
            raise ValueError(f"No columns found for table '{self.table_name}'.")

    def _to_list_of_lists(self, records: list[Any]) -> list[list[Any]]:
        if not records:
            return []
        return [list(record._mapping.values()) for record in records]

    def _to_single_dict(self, records: list[Any]) -> dict[str, list[Any]]:
        if not records:
            return {}
        result = {key: [] for key in records[0]._mapping.keys()}
        for record in records:
            for key, value in record._mapping.items():
                result[key].append(value)
        return result


class OhlcvTable(BaseTable):
    def __init__(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        db_manager: "DatabaseManager",
        repo: Coroutine = None
    ):
        super().__init__(db_manager, repo)
        self.exchange = exchange
        self.symbol = "".join(symbol.split("/"))
        self.interval = interval
        self.table_name = f"{self.exchange}_{self.symbol}_{self.interval}"
        logger.info(f"Creating OHLCV table: {self.table_name} for {exchange} {symbol}")

    async def exists(self) -> bool:
        """Checks if the OHLCV table exists in the database."""
        query = f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = '{self.table_name}'
        """
        result = await self.db.fetch_one(query)
        return result[0] > 0

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

        logger.info(
            "Table %s created: %s",
            self.table_name, "OK" if await self.exists() else "FAIL"
        )

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

    # ................................................................................
    async def fetch_latest(self, limit: int = 1000) -> list[list[Any]]:
        if await self.needs_update():
            await self.update()

        query = f"""
        SELECT * FROM {self.table_name}
        ORDER BY openTime DESC
        LIMIT :limit
        """
        result = await self.db.fetch_all(query, values={"limit": limit})
        result = reversed(self._to_list_of_lists(result))
        return Ohlcv(
            exchange=self.exchange,
            symbol=self.symbol,
            interval=self.interval,
            data=result
        )

    async def fetch_by_range(self, start: int, end: int) -> list[list[Any]]:
        if await self.needs_update(end):
            await self.update(end)

        query = f"""
        SELECT * FROM {self.table_name}
        WHERE openTime BETWEEN :start AND :end
        ORDER BY openTime ASC
        """
        result = await self.db.fetch_all(query, {"start": start, "end": end})
        result = reversed(self._to_list_of_lists(result))

        return Ohlcv(
            exchange=self.exchange,
            symbol=self.symbol,
            interval=self.interval,
            start=start,
            end=end,
            data=result
        )

    # ................................................................................
    async def get_row_count(self) -> int:
        """
        Returns the number of rows in the table.
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        result = await self.db.fetch_one(query)
        return result[0] if result else 0

    async def get_first_entry_ts(self) -> int:
        """ Returns the timestamp of the first entry in the table."""
        query = f"SELECT MIN(openTime) FROM {self.table_name}"
        result = await self.db.fetch_one(query)
        return result[0] if result else 0

    async def get_last_entry_ts(self) -> int:
        """
        Returns the timestamp of the last entry in the table.
        """
        query = f"SELECT MAX(openTime) FROM {self.table_name}"
        result = await self.db.fetch_one(query)
        return result[0] if result else 0

    # ................................................................................
    async def needs_update(self, up_to: int = None) -> bool:
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
        now_ms = up_to or datetime.datetime.now().timestamp() * 1000
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

        logger.info(
            "%s needs an update: %s  (now_utc: %s, latest_close_utc: %s, "
            "time delta: %s, interval: %s)",
            self.table_name, needs_update, now_utc, latest_close_utc,
            seconds_to(delta / 1000), seconds_to(interval_ms / 1000)
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
        if end is None:
            end = datetime.datetime.now().timestamp() * 1000

        start = await self.get_last_entry_ts() + 1

        logger.info(
            "Updating %s from %s to %s",
            self.table_name,
            datetime
            .datetime
            .fromtimestamp(start / 1000, datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S"),
            datetime
            .datetime
            .fromtimestamp(end / 1000, datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S"),
            )

        request = {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "start": int(start),
            "end": int(end),
        }

        response = await process_request(request)

        if not response:
            logger.error(f"No data received for {self.table_name}")
            return

        if not response.success:
            logger.error(
                "Error fetching data from %s: %s", self.table_name, response.error
                )
            return

        logger.info(f"Received {len(response.data)} rows for {self.table_name}")
        await self.insert(response.data[:-1])


class DatabaseManager:
    def __init__(self):
        db_user = os.getenv("DB_USER")
        db_password = str(os.getenv("DB_PASS"))
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "akasha")

        if not db_user or not db_password:
            raise EnvironmentError(
                "DB_USER and DB_PASSWORD environment variables must be set."
                )

        # Encode the username
        encoded_user = quote_plus(db_user)

        self._db_url = (
            f"mysql+aiomysql://{encoded_user}:{db_password}@{db_host}"
            f":{db_port}/{db_name}"
        )
        logger.debug(f"Database URL: {self._db_url}")
        self._database = Database(self._db_url)

    async def connect(self):
        try:
            await self._database.connect()
        except Exception as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False
        logger.debug("✅ Connected to the MySQL database.")
        return True

    async def disconnect(self):
        await self._database.disconnect()
        logger.debug("🛑 Disconnected from the MySQL database.")

    @property
    def db(self):
        return self._database

    async def get_table(self, exchange: str, symbol: str, interval: str) -> OhlcvTable:
        return OhlcvTable(exchange, symbol, interval, self)