#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fir Dec 20 10:20:33 2024

@author dhaneor
"""
import datetime
import logging
import pandas as pd
import time

from src.data.util.mnemosyne import Mnemosyne
from src.exchange.binance_ import Binance
from src.exchange.kucoin_ import KucoinCrossMargin as Kucoin
from util.timeops import (
    unix_to_utc,
    utc_to_unix,
)

# ==============================================================================
# dictionary of interval names and their lengths in seconds
INTERVAL_LENGTHS = {
    '1m': 60_000,
    '5m': 300_000,
    '15m': 900_000,
    '30m': 1_800_000,
    '1h': 3_600_000,
    '2h': 7_200_000,
    '4h': 14_400_000,
    '6h': 21_600_000,
    '12h': 43_200_000,
    '1d': 86_400_000,
    '1w': 604_800_000
}

VALID_EXCHANGES = {
    "binance": {
        "api_class": Binance,
        "exchange_opened": utc_to_unix("July 01, 2017 00:00:00"),
        "max_workers": 5,
    },
    "kucoin": {
        "api_class": Kucoin,
        "exchange_opened": utc_to_unix("September 27, 2017 00:00:00"),
        "max_workers": 2,
    },
}

MAX_WORKERS_BINANCE = VALID_EXCHANGES["binance"]["max_workers"]
MAX_WORKERS_KUCOIN = VALID_EXCHANGES["kucoin"]["max_workers"]

logger = logging.getLogger('main.ohlcv')


class OHLCVData:
    def __init__(self, symbol: str, data: pd.DataFrame | None):
        self.symbol = symbol
        self.data = data

    def _determine_interval(self) -> str:
        """Finds the interval of the OHLCV data."""

        # make a list of intervals from the data, and remove the
        # first element (which is NaT)
        intervals = self.data['open time'].diff().tolist()[1:]

        # find the interval with the smallest difference in seconds
        interval = min(intervals, key=lambda x: abs(x.total_seconds())).total_seconds()

        return next(
            (
                interval_name for interval_name, interval_length
                in INTERVAL_LENGTHS.items()
                if interval_length == interval
                ),
            'unknown'
            )


class OHLCVTable:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

    @property
    def name(self) -> str:
        symbol = self.symbol.replace("-", "") if "-" in self.symbol else self.symbol
        return "_".join([self.exchange_name, symbol, "ohlcv", self.interval])

    @property
    def exchange_name(self) -> str:
        return 'kucoin' if '-' in self.symbol else 'binance'

    def data(self, start: int, end: int, attempt: int = 0) -> pd.DataFrame | None:
        """Gets OHLCV data from the database."""

        attempt += 1

        if attempt == 3:
            logger.error(f"Failed to fetch data after {attempt - 1} attempts")
            return None

        # fetch data from database (returns dataframe)
        logger.info(f"[{attempt}] fetching data from database...")

        if not self._ohlcv_table_exists:
            self.create(self.name)
            self.update(self.name)

        with Mnemosyne() as conn:
            data = conn.get_ohclv(
                exchange=self.exchange_name,
                symbol=self.symbol,
                interval=self.interval,
                start=start,
                end=end,
            )

        # happy case
        if data is not None:
            logger.debug(f"got {len(data)} rows of data")

            # check if we got only partial data and need to update the table
            latest_open = int(data.at[data.last_valid_index(), "open time"])
            latest_close = latest_open + INTERVAL_LENGTHS[self.interval]
            # now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            update_necessary = latest_close <= end

            if not update_necessary:
                return data

            # update the table and call this method again
            self.update()
            return self.data(start, end, attempt)

        # unhappy case (the database does not contain any data for
        # this period). this can happen if the table has not been updated for a
        # long time and the requested data period starts after the
        # date/time of the last update
        logger.warning("failed to fetch data from database - going to update table...")

        # try to update the table again
        self.update()
        return self.data(start, end, attempt)

    def create(self, table_name: str):
        table_description = (
            """
                        CREATE TABLE IF NOT EXISTS `%s` (
                        id INTEGER PRIMARY KEY AUTO_INCREMENT,
                        openTime VARCHAR(15) NOT NULL UNIQUE,
                        humanOpenTime VARCHAR(32) NOT NULL,
                        open VARCHAR(32) NOT NULL,
                        high VARCHAR(32) NOT NULL,
                        low VARCHAR(32) NOT NULL,
                        close VARCHAR(32) NOT NULL,
                        volume VARCHAR(32) NOT NULL,
                        closeTime BIGINT NOT NULL,
                        quoteAssetVolume VARCHAR(32) NOT NULL,
                        numberOfTrades INT,
                        takerBuyBaseAssetVolume VARCHAR(32),
                        takerBuyQuoteAssetVolume VARCHAR(32)
                        ) """
            % table_name
        )

        with Mnemosyne() as conn:
            conn.create_table(table_description)

    def update(self) -> bool:
        """Update OHLCV table up to current date/time.

        This method checks the table status, especially the most recent
        entry and (if necessary) updates the table.
        """
        # get table status from database
        status = self.status

        logger.debug(f"table status: {status}")

        # if the table is empty we need to set a sane value (around the
        # time the exchange opened) for 'latest open', from where the
        # download of new data will start
        latest_open = status.get("latest open", 0)
        latest_open_utc = unix_to_utc(latest_open)

        if latest_open == 0:
            latest_open = VALID_EXCHANGES\
                .get(self.exchange_name, {})\
                .get("exchange_opened", 0)

        # check if the latest row in table already represents the most
        # recent datapoint and return if true
        if self.is_up_to_date(latest_open, self.interval):
            logger.debug(
                f"no update necessary for {self.symbol} - {self.interval} "
                f"(latest open time: {latest_open_utc})"
            )
            return True

        logger.info(
            f"updating {self.name} from {latest_open_utc} ({latest_open}) "
            f"to now for {self.symbol} (interval {self.interval})"
        )

        # get ohlcv data from exchange
        try:
            with self.exchange() as conn:
                res = conn.get_ohlcv(
                    symbol=self.symbol,
                    interval=self.interval,
                    start=latest_open + 1,
                    end=int(time.time()) * 1000,
                    as_dataframe=False,
                )

            if not res.get("success", False):
                raise Exception(res.get("error"))

            msg = res.get("message", None)

            if msg is None:
                logger.error(res)
                raise Exception('update succeeded, but "message" is empty')

            logger.debug(f"got {len(msg)} rows of data from the API")

        except Exception as e:
            logger.error(f"update failed for {self.name}: {self.exchange_name}: {e}")
            return False

        self.write(table_name=self.name, data=msg)

    def write(
        self, table_name: str, data: list[list], end_of_table=True
    ) -> bool:
        """Writes OHLCV data to table in database.

        :param table_name: name of table to write to
        :type table_name: str
        :param data: raw ohlcv data as downloaded from exchange
        :type data: list
        :param end_of_table: set to False for filling in missing values
        (see below), defaults to True
        :type end_of_table: bool, optional
        :return: successful or not
        :rtype: bool
        """
        if not data:
            logger.warning(
                f"someone tried to update a table, but data is empty ({data})"
            )
            return False

        columns = [
            "openTime",
            "humanOpenTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "quoteAssetVolume",
            "numberOfTrades",
            "takerBuyBaseAssetVolume",
            "takerBuyQuoteAssetVolume",
        ]

        max_simultaneous, sim_counter, save_data = 10000, 0, []

        # NOTE : usually the downloaded data includes the current
        # (unfinished) period, which would insert incorrect data
        # into the database.
        # Therefore we can delete this - most of the time!
        #
        # When updating the table with missing data, this must not be
        # deleted. The behaviour can be set with the 'end_of_table'
        # flag.
        if end_of_table:
            del data[-1]

        logger.debug(f"saving {len(data)} items to {self.name}")

        # prepare the data row by row and do a batch save to database
        for index, row in enumerate(data):
            sim_counter += 1
            if not isinstance(row, list):
                logger.error(f"row {index} in {self.name} is not a list: {row}")
                continue

            # kline format from Kucoin is different, so we need to convert
            # to match the Binance format
            if self.exchange_name == "kucoin":
                row = self._standardize_kucoin_kline(row, self.interval)

            # insert human readable time as item in list/row
            try:
                row.insert(1, unix_to_utc(row[0]))
            except Exception as e:
                logger.error(
                    f"unable to insert human readable time in row {index}: {e}"
                )
                return False

            # delete the last value from the row, it's not needed
            del row[-1]

            save_data.append(tuple(row))

            # save to database when number of datasets equals value
            # for max_simultaneous (batch write) or we reached the
            # end of our list
            if sim_counter == max_simultaneous or index == (len(data) - 1):
                logger.debug(f"saving batch ({len(save_data)}) to {self.name}")

                with Mnemosyne() as conn:
                    conn.save_to_database(self.name, columns, save_data, batch=True)

                save_data, sim_counter = [], 0

        logger.info(f"update of {self.name} successful")
        return True

    @property
    def status(self) -> dict:
        """Get the table status for a specific symbol.

        :param symbol: name of the symbol
        :type symbol: str
        :param interval: name of the (kline) interval '15m', '30m' ... '1d'
        :return: the informations about the table status (see above)
        :rtype: dict

        .. code:: python
            {
                'symbol' : symbol
                'interval' : interval
                'name' : table_name,
                'rows' : no_of_rows,
                'earliest open' : earliest timestamp 'open time',
                'latest open' : latest timestamp 'open time',
                'latest utc' : latest 'open time' in UTC
            }
        """
        with Mnemosyne() as conn:
            status = conn.get_ohlcv_table_status(self.name)

        if status:
            status["symbol"], status["interval"] = self.symbol, self.interval

        return status

    def _get_symbol_and_interval_from(self, table_name: str) -> tuple:
        interval = table_name.split("_")[-1]

        if self.exchange_name == "binance":
            _parts = table_name.split("_")
            return _parts[1], _parts[3]

        if self.exchange_name == "kucoin":
            tradebale = self.get_tradeable_symbols()
            if tradebale:
                return [
                    s
                    for s in tradebale
                    if s.split("-")[0] in table_name and s.split("-")[1] in table_name
                ][0], interval
            else:
                logger.error(
                    f'could not determine interval from tablename "{table_name}" '
                    f"- writing to table aborted"
                )
                return None, None

    def _ohlcv_table_exists(self, symbol: str, interval: str) -> bool:
        with Mnemosyne() as conn:
            exists = conn.check_if_table_exists(self.name)
        return exists

    def is_up_to_date(self) -> bool:
        interval_in_ms = INTERVAL_LENGTHS[self.interval]
        now = datetime.datetime.now(datetime.UTC).timestamp() * 1000
        latest_open = self.status.get("latest open")

        if not latest_open:
            logger.error(f"No table in database for: {self.name}")
            return False

        now_utc = (
            datetime
            .datetime
            .fromtimestamp(now / 1000, datetime.UTC)
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        latest_open_utc = (
            datetime
            .datetime
            .fromtimestamp(latest_open / 1000, datetime.UTC)
            .strftime("%Y-%m-%d %H:%M:%S")
            )

        is_up_to_date = latest_open > now - interval_in_ms

        logger.debug(
            "latest_open_utc: %s, now_utc: %s, interval: %s, is_up_to_date: %s",
            latest_open_utc,
            now_utc,
            self.interval,
            is_up_to_date,
        )
        return is_up_to_date

