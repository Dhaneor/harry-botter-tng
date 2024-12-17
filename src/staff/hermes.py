#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:57:06 2021

@author: dhaneor
"""
import datetime
import inspect
import sys
import concurrent.futures
import time
import pandas as pd
import logging

from pprint import pprint
from functools import lru_cache
from typing import Union, Tuple, Optional, Callable

from src.staff.mnemosyne import Mnemosyne
from src.exchange.binance_ import Binance
from src.exchange.kucoin_ import KucoinCrossMargin as Kucoin
from util.timeops import (
    time_difference_in_ms,
    unix_to_utc,
    interval_to_milliseconds,
    get_start_and_end_timestamp,
    utc_to_unix,
    seconds_to,
)
from src.helpers.ilabrat import get_exchange_name


# ==============================================================================
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

logger = logging.getLogger("main.hermes")
logger.setLevel(logging.INFO)


# ==============================================================================
class BadSymbolError(Exception):
    """Raised when a symbol is not found in the symbols table."""


# ==============================================================================
"""
TODO    when 'market' is set to CROSS MARGIN (or ISOLATED MARGIN in
        the future), get_tradeable_symbols should filter for MARGIN.
        Right now, all SPOT symbols are returned ... doesn't really
        matter for now, but should be corrected!

TODO    rewrite everything! Should be two different classes, each
        representing a repository: one for symbols and one for OHLCV
        data. Should also use the same class for all exchanges, with
        CCXT in the background.

        Also: Hermes and HermesDataBase should share the same class.
        Right now, they are both using methods of each other ... this
        is weird!

        Also: make these new classes async!

TODO    implement a solution for missing periods in the OHLCV data!
"""


# ==============================================================================
class HermesDataBase:
    """This class handles all database operations for Hermes.

    This is a base class and should not be used on its own, but only as
    part of the Hermes class!
    """

    def __init__(self, exchange_name: str, exchange_obj):
        self.exchange_name: str = exchange_name
        self.exchange = exchange_obj
        self.updateAfter = 14400  # seconds for symbols table update interval
        self._max_workers = 1  # max workers to use for database update

    # --------------------------------------------------------------------------
    def update_database_for_interval(
        self, exchange: str, quote_asset: str, interval: str
    ) -> None:
        self._set_exchange(exchange)

        # get list of all currently tradeable (=active and not
        # restricted) symbols on 'exchange' for the given 'quote_asset'
        tradebale = self.get_tradeable_symbols(quote_asset=quote_asset)

        # create all table names that should be in our database
        table_names = [
            self._get_ohlcv_table_name(sym, interval) for sym in tradebale
            ]

        # get a list of all symbols that we have in our database for
        # the given interval
        with Mnemosyne() as conn:
            _all_in_db = conn.get_all_table_names(containing=self.exchange_name)

        # compare the two lists and create all missing tables
        _no_table_for = [tn for tn in table_names if tn not in _all_in_db]

        # create the missing tables
        [self._create_ohlcv_table(tn) for tn in _no_table_for]

        # ......................................................................
        # update all tables that are not up to date
        #
        # prepare the exchange connection and determine the number of
        # workers (threads to be used)
        _results = []
        self.conn = self.exchange()
        if self.conn.client is None:
            logger.debug(f"no connection to {self.exchange_name}")
            return

        self._max_workers, conn_max_workers = self._get_max_workers(interval)
        self.conn.set_max_workers(conn_max_workers)
        self.conn.verbose = False

        if self.exchange_name == "kucoin":
            self.conn.delay = 2 if interval in ["15m", "30m", "1h"] else 1

        logger.debug(f"{self._max_workers=} {conn_max_workers=}")
        # time.sleep(5)

        # start the update(s) with parallel threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures = []
            for tn in table_names:
                futures.append(executor.submit(self._update_ohlcv_table, table_name=tn))

            futures, _ = concurrent.futures.wait(futures)
            for future in futures:
                _results.append(future.result())

    def check_table(self, symbol: str, interval: str) -> Tuple[list[dict]]:
        if symbol not in self.tradeable_symbols:
            logger.error(f"{symbol} not found on {self.exchange_name.upper()}!")
            return

        table_exists = self._ohlcv_table_exists(symbol, interval)
        logger.debug(f"table exists: {table_exists}")

        table_name = self._get_ohlcv_table_name(symbol, interval)

        # create the table from scratch if it doesn't exist yet
        if not table_exists:
            self._create_ohlcv_table(table_name)

            if self._ohlcv_table_exists(symbol, interval):
                self._update_ohlcv_table(symbol, interval)

        status = self._get_ohlcv_table_status_for_symbol(symbol, interval)
        logger.debug(status)

        # determine start and end time for the ohlcv data for the given symbol
        start = status.get("earliest open", 0)
        end = int(time.time() * 1000)

        if start is None:
            start = VALID_EXCHANGES[self.exchange_name]["exchange_opened"]

        start_human = f"{unix_to_utc(start)} ({start})"
        end_human = f"{unix_to_utc(end)} ({end})"

        logger.debug(f"fetching data from {start_human} to {end_human} from DB ...\n")

        # get the data from the database
        with Mnemosyne() as conn:
            data = conn.get_ohclv(
                exchange=self.exchange_name,
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )

        if data is None or isinstance(data, dict):
            raise Exception(f"failed to fetch data from {self.exchange_name.upper()}!")

        # find missing periods in the data
        if not data.empty:
            missing_periods = self._find_missing_rows_in_df(data)
            no_of_missing_periods = len(missing_periods)

            if no_of_missing_periods == 0:
                logger.info("Everything all right with this table, Sir!")
                return
        else:
            logger.error("No data found in the database!")
            raise Exception(
                "no data found in the database - unable to find missing periods"
            )

        logger.debug("-~•~-" * 40)
        logger.debug(f"found {no_of_missing_periods} MISSING PERIODS in the table")

        successful, failed, still_missing = 0, 0, []

        # try to download missing period data from API
        with self.exchange() as conn:
            for idx, period in enumerate(missing_periods):
                logger.info(
                    f"missing period {idx}/{no_of_missing_periods}:"
                    f" {period[-3]} - {period[-2]} ({period[-1]}) ::"
                    f" {period[-5]} - {period[-6]}"
                )

                start, end = period[0], period[1] - 1

                # try to download missing period data from API

                res = conn.get_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start=int(period[0]),
                    end=int(period[1]) - 1,
                    as_dataframe=False,
                )

                if not res.get("success"):
                    error = res.get("error")
                    logger.error(f"FAIL: {error}")
                    failed += 1
                    continue

                data = res.get("message", [])
                data = [elem for elem in data if elem[0] < end]

                # if this worked and we have data, write it to the database.
                # this means that something went wrong during creation or
                # a previous update of this table, and we can now complete
                # the data.
                if data:
                    logger.info(
                        f"SUCCESS: got ({len(data)} rows of data for "
                        f"period {idx}/{no_of_missing_periods}"
                    )

                    self._write_to_ohlcv_table(table_name, data, False)
                    successful += 1

                # if this worked, but we don't have data, log an error
                # this happens frequently, because for missing periods, ususally
                # there just is not data available
                else:
                    logger.info(
                        f"FAIL: got no data for period {idx}/{no_of_missing_periods}"
                    )
                    failed += 1
                    still_missing.append(idx)

            return {
                "missing_periods": missing_periods,  # previously missing
                "still_missing": still_missing,  # indexes of still missing periods
                "successful": successful,  # number of successful downloads
                "failed": failed,  # number of failed downloads
            }

    # --------------------------------------------------------------------------------
    def _find_missing_rows_in_df(self, df: pd.DataFrame) -> tuple:
        if len(df) < 2:
            logger.warning(f"dataframe too short ({len(df)} to anaylze!)")
            return tuple()

        for idx in range(len(df) - 1):
            df["time_diff"] = df["open time"].diff()
            interval_in_ms = df["open time"].diff().median()

            df.loc[df.time_diff > interval_in_ms, "missing_from"] = (
                df["open time"].shift() + interval_in_ms - 1
            )
            df.loc[df.time_diff > interval_in_ms, "missing_to"] = df["open time"] - 1

            missing_from = df.loc[(df.time_diff > interval_in_ms)][
                "missing_from"
            ].tolist()
            missing_to = df.loc[(df.time_diff > interval_in_ms)]["missing_to"].tolist()
            duration = df.loc[(df.time_diff > interval_in_ms)]["time_diff"].tolist()

            df.drop("time_diff", axis=1, inplace=True)

            return tuple(
                zip(
                    missing_from,
                    missing_to,
                    duration,
                    map(unix_to_utc, missing_from),
                    map(unix_to_utc, missing_to),
                    map(seconds_to, map(lambda x: x / 1000, duration)),
                )
            )

    def _find_maintenance_windows(self, interval: str):
        self.interval = interval
        self._find_lookback(self.interval)

        symbols = ["ADAUSDT"]
        all_ = []

        for s in symbols:
            self.hermes.set_symbol(s)
            self.symbol_name = s
            all_ += self._find_missing_rows_in_df()

        counted = {x: all_.count(x) for x in all_}
        m_windows = [k for k, v in counted.items() if v == len(symbols)]

        logger.debug(f"found the following downtimes based on {interval} interval:")
        pprint(m_windows)

    # --------------------------------------------------------------------------------
    # methods that deal with the ohlcv tables
    def _get_ohlcv_from_database(
        self, symbol: str, interval: str, start: int, end: int, attempt: int = 0
    ) -> pd.DataFrame | None:
        """Gets OHLCV data from database."""

        attempt += 1

        if attempt == 3:
            logger.error(f"Failed to fetch data after {attempt - 1} attempts")
            return None

        # fetch data from database (returns dataframe)
        logger.info(f"[{attempt}] fetching data from database...")

        if not self._ohlcv_table_exists(symbol, interval):
            table_name = self._get_ohlcv_table_name(symbol, interval)
            self._create_ohlcv_table(table_name)
            self._update_ohlcv_table(table_name)

        with Mnemosyne() as conn:
            data = conn.get_ohclv(
                exchange=self.exchange_name,
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )

        # happy case
        if data is not None:
            logger.debug(f"got {len(data)} rows of data")

            # check if we got only partial data and need to update the table
            latest_open = int(data.at[data.last_valid_index(), "open time"])
            last_close = latest_open + interval_to_milliseconds(interval)
            now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            update_necessary = last_close + interval_to_milliseconds(interval) < now

            if not update_necessary:
                return data

            # update the table and call this method again
            self._update_ohlcv_table(symbol=symbol, interval=interval)
            return self._get_ohlcv_from_database(
                symbol, interval, start, end, attempt
                )

        # unhappy case (the database does not contain any data for
        # this period). this can happen if the table has not been updated for a
        # long time and the requested data period starts after the
        # date/time of the last update
        logger.debug("failed to fetch data from database - going to update table...")

        # try to update the table again
        self._update_ohlcv_table(symbol=symbol, interval=interval)
        return self._get_ohlcv_from_database(symbol, interval, start, end, attempt)

    def _create_ohlcv_table(self, table_name: str):
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

    def _update_ohlcv_table(
        self,
        table_name: Optional[str] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> bool:
        """Update OHLCV table up to current date/time.

        This method checks the table status, especially the most recent
        entry and (if necessary) updates the table. It can be called
        with 'table_name' alone or with 'symbol' and 'interval' (both
        of them must be provided then).

        :param table_name: table name to be updated, defaults to None
        :type table_name: str, optional
        :param symbol: symbol, defaults to None
        :type symbol: str, optional
        :param interval: interval, defaults to None
        :type interval: str, optional
        :raises ValueError: ValueError if not enough parameters
        :return: True if update was successful or otherwise False
        :rtype: bool
        """
        # add parameter(s) symbol/interval or table_name depending on
        # parameters provided
        if table_name:
            symbol, interval = self._get_symbol_and_interval_from(table_name)
        else:
            if symbol is None or interval is None:
                raise ValueError(
                    "Please specify 'table_name' <or> 'symbol' and 'interval'"
                    )
            table_name = self._get_ohlcv_table_name(symbol, interval)

        # get table status from database
        status = self._get_ohlcv_table_status_for_symbol(symbol, interval)
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
        if not self._ohlcv_table_needs_update(latest_open, interval):
            logger.debug(
                f"no update necessary for {symbol} - {interval} "
                f"(latest open time: {latest_open_utc})"
            )
            return True

        logger.info(
            f"updating {table_name} from {latest_open_utc} ({latest_open}) "
            f"to now for {symbol} (interval {interval})"
        )

        # get ohlcv data from exchange
        try:
            with self.exchange() as conn:
                res = conn.get_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start=latest_open + 1,
                    end=int(time.time()),
                    as_dataframe=False,
                )

            if not res.get("success", False):
                raise Exception(res.get("error"))

            msg = res.get("message", None)

            if msg is None:
                logger.error(res)
                raise Exception('update succeeded, but "message" is empty')

        except Exception as e:
            logger.error(f"update failed for {table_name}: {self.exchange_name}: {e}")
            return False

        self._write_to_ohlcv_table(table_name=table_name, data=msg)

    def _write_to_ohlcv_table(
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
        _, interval = self._get_symbol_and_interval_from(table_name)

        if not interval:
            logger.error(f"unable to determine interval for {table_name}")
            return False

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

        logger.debug(f"saving {len(data)} items to {table_name}")

        if len(data) == 1:
            logger.debug(data)
            logger.debug(f"skipping saving of 1 item to {table_name}")
            return True

        # prepare the data row by row and do a batch save to database
        # for index in trange(len(data), unit=' items', desc=table_name):
        for index, row in enumerate(data):
            sim_counter += 1
            if not isinstance(row, list):
                logger.error(f"row {index} in {table_name} is not a list: {row}")
                continue

            # kline format from Kucoin is different, so we need to convert
            # to match the Binance format
            if self.exchange_name == "kucoin":
                row = self._standardize_kucoin_kline(row, interval)

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

            # logger.debug(f"row {index}: {row}")

            save_data.append(tuple(row))

            # save to database when number of datasets equals value
            # for max_simultaneous (batch write) or we reached the
            # end of our list
            if sim_counter == max_simultaneous or index == (len(data) - 1):
                logger.debug(f"saving batch ({len(save_data)}) to {table_name}")

                with Mnemosyne() as conn:
                    conn.save_to_database(table_name, columns, save_data, batch=True)

                save_data, sim_counter = [], 0

            # else:
            #     logger.debug(
            #         f"{index=}: save_data has {len(save_data)} items now :: "
            #         f"the end is near: {index == (len(data) - 1)}"
            #     )

        logger.info(f"update of {table_name} successful")
        return True

    def _get_ohlcv_table_status_for_symbol(self, symbol: str, interval: str) -> dict:
        """Get the table status for a specific symbol.

        :param symbol: name of the symbol
        :type symbol: str
        :param interval: name of the (kline) interval '15m', '30m' ... '1d'
        :return: the informations about the table status (see above)
        :rtype: dict

        .. code:: python
            {
                'name' : table_name,
                'table exists' : True|False,
                'rows' : no_of_rows,
                'earliest open' : earliest timestamp 'open time',
                'latest open' : latest timestamp 'open time',
                'symbol' : symbol
                'interval' : interval
            }
        """
        table_name = self._get_ohlcv_table_name(symbol=symbol, interval=interval)

        with Mnemosyne() as conn:
            status = conn.get_ohlcv_table_status(table_name)

        if status:
            status["symbol"], status["interval"] = symbol, interval

        return status

    def _get_ohlcv_table_name(self, symbol: str, interval: str) -> str:
        exchange = get_exchange_name(symbol)
        symbol = symbol.replace("-", "") if "-" in symbol else symbol
        return "_".join([exchange, symbol, "ohlcv", interval])

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
        table_name = self._get_ohlcv_table_name(symbol, interval)
        all_tables = self._get_alltable_names()
        return True if table_name in all_tables else False

    def _ohlcv_table_needs_update(self, latest_open: int, interval: str) -> bool:
        interval_in_ms = interval_to_milliseconds(interval)
        now = datetime.datetime.now(datetime.UTC).timestamp() * 1000
        return latest_open + 2 * interval_in_ms < now

    # --------------------------------------------------------------------------
    # methods that deal with the symbol(s) information table(s)
    def _create_symbols_table(self, table_name: str) -> bool:
        table_description = (
            """
            CREATE TABLE IF NOT EXISTS `%s`(
                baseAsset VARCHAR(16) NOT NULL,
                baseAssetPrecision TINYINT NOT NULL,
                baseCommissionPrecision TINYINT NOT NULL,
                cancelReplaceAllowed BOOL,
                f_icebergParts_limit VARCHAR(8) NOT NULL,
                f_lotSize_maxQty VARCHAR(32) NOT NULL,
                f_lotSize_minQty VARCHAR(32) NOT NULL,
                f_lotSize_stepSize VARCHAR(16) NOT NULL,
                f_marketLotSize_maxQty VARCHAR(32) NOT NULL,
                f_marketLotSize_minQty VARCHAR(32) NOT NULL,
                f_marketLotSize_stepSize VARCHAR(16) NOT NULL,
                f_maxNumOrders VARCHAR(8) NOT NULL,
                f_maxNumAlgoOrders VARCHAR(8) NOT NULL,
                f_minNotional_applyToMarket BOOL NOT NULL,
                f_minNotional_avgPriceMins VARCHAR(8) NOT NULL,
                f_minNotional_minNotional VARCHAR(16) NOT NULL,
                f_percentPrice_avgPriceMins VARCHAR(8) NOT NULL,
                f_percentPrice_multiplierDown VARCHAR(8) NOT NULL,
                f_percentPrice_multiplierUp VARCHAR(8) NOT NULL,
                f_priceFilter_maxPrice VARCHAR(32) NOT NULL,
                f_priceFilter_minPrice VARCHAR(32) NOT NULL,
                f_priceFilter_tickSize VARCHAR(16) NOT NULL,
                f_trailingDelta_maxTrailingAboveDelta VARCHAR(16),
                f_trailingDelta_maxTrailingBelowDelta VARCHAR(16),
                f_trailingDelta_minTrailingAboveDelta VARCHAR(16),
                f_trailingDelta_minTrailingBelowDelta VARCHAR(16),
                icebergAllowed BOOL NOT NULL,
                isSpotTradingAllowed BOOL NOT NULL,
                isMarginTradingAllowed BOOL NOT NULL,
                ocoAllowed BOOL NOT NULL,
                orderTypes VARCHAR(256) NOT NULL,
                permissions VARCHAR(32) NOT NULL,
                quoteAsset VARCHAR(16) NOT NULL,
                quoteAssetPrecision TINYINT NOT NULL,
                quoteCommissionPrecision TINYINT NOT NULL,
                quoteOrderQtyMarketAllowed BOOL NOT NULL,
                quotePrecision TINYINT NOT NULL,
                status VARCHAR(32) NOT NULL,
                symbol VARCHAR(16) PRIMARY KEY UNIQUE,
                updateTime INT NOT NULL
                ) """
            % table_name
        )

        with Mnemosyne() as conn:
            conn.create_table(table_description)

            if conn.check_if_table_exists(table_name):
                logger.debug(f"created symbols table for {self.exchange_name}")
                return True
            else:
                logger.debug(f"table {table_name} could not be created")
                return False

    def _is_symbols_table_update_necessary(self, exchange_name: str) -> bool:
        """Checks if we need to update our symbols table for the given exchange.

        :param exchange_name: 'binance' or 'kucoin'
        :type exchange_name: str
        :raises ValueError: raises error if exchange_name is invalid
        :return: True if the last update is older than self.updateAfter (sec)
        :rtype: bool
        """
        if exchange_name == "kucoin":
            symbol = "BTC-USDT"
        elif exchange_name == "binance":
            symbol = "BTCUSDT"
        else:
            raise ValueError(f"{exchange_name} is not a valid exchange")

        test = self._get_symbol_from_database(symbol)

        if test:
            updateTime = test.get("updateTime", 0)
            if time.time() - updateTime > self.updateAfter:
                return True

        return False

    def _update_symbols_table(self, exchange_name: str) -> bool:
        """Updates the symbols table with the symbol(s) information.

        :returns: True if successful else False
        :rtype: bool
        """
        symbols = self._get_all_symbols_from_exchange(exchange_name)

        if symbols is not None:
            table_name = f"{exchange_name}_symbols"
            self._drop_symbols_table(table_name)

            if self._create_symbols_table(table_name):
                self._write_to_symbols_table(table_name=table_name, data=symbols)
            else:
                raise Exception("could not create symbols table for update")
        else:
            logger.debug("[_update_symbols_table] unable to create symbols table!")
            logger.debug(f"this is what we got from the exchange: {type(symbols)}")

    def _write_to_symbols_table(self, table_name: str, data: list) -> None:
        """Writes the symbols(s) information to our database.

        :param table_name: name if the table that contains the symbol info
        :type table_name: str
        :param data: the symbol(s) info as given by our exchange class(es)
        :type data: list

        .. code:: python

            {
                'baseAsset': 'WOO',
                'baseAssetPrecision': 8,
                'baseCommissionPrecision': 8,
                'filters': [{'filterType': 'PRICE_FILTER',
                            'maxPrice': '1000.00000000',
                            'minPrice': '0.00000001',
                            'tickSize': '0.00000001'},
                            {'avgPriceMins': 5,
                            'filterType': 'PERCENT_PRICE',
                            'multiplierDown': '0.2',
                            'multiplierUp': '5'},
                            {'filterType': 'LOT_SIZE',
                            'maxQty': '92141578.00000000',
                            'minQty': '0.10000000',
                            'stepSize': '0.10000000'},
                            {'applyToMarket': True,
                            'avgPriceMins': 5,
                            'filterType': 'MIN_NOTIONAL',
                            'minNotional': '0.00010000'},
                            {'filterType': 'ICEBERG_PARTS', 'limit': 10},
                            {'filterType': 'MARKET_LOT_SIZE',
                            'maxQty': '66126.34611111',
                            'minQty': '0.00000000',
                            'stepSize': '0.00000000'},
                            {'filterType': 'MAX_NUM_ORDERS', 'maxNumOrders': 200},
                            {'filterType': 'MAX_NUM_ALGO_ORDERS', 'maxNumAlgoOrders': 5}
                            ],
                'icebergAllowed': True,
                'isMarginTradingAllowed': False,
                'isSpotTradingAllowed': True,
                'ocoAllowed': True,
                'orderTypes': ['LIMIT',
                                'LIMIT_MAKER',
                                'MARKET',
                                'STOP_LOSS_LIMIT',
                                'TAKE_PROFIT_LIMIT'],
                'permissions': ['SPOT'],
                'quoteAsset': 'BTC',
                'quoteAssetPrecision': 8,
                'quoteCommissionPrecision': 8,
                'quoteOrderQtyMarketAllowed': True,
                'quotePrecision': 8,
                'status': 'TRADING',
                'symbol': 'WOOBTC'
            }
        """
        _flattened = [self._flatten_symbol_dictionary(item) for item in data]
        columns = list(_flattened[0].keys())
        values = [tuple(item.values()) for item in _flattened]

        for v in values:
            if len(v) > 40:
                raise Exception(f"item unfit for saving to database (len: {len(v)})")

        with Mnemosyne() as conn:
            conn.save_to_database(table_name, columns, values, batch=True)

    def _drop_symbols_table(self, table_name: str) -> bool:
        try:
            with Mnemosyne() as conn:
                conn.drop_table(table_name)
                logger.debug(f"Successfully deleted table {table_name}")
                return True
        except Exception as e:
            logger.debug(e)
            return False

    # ---------------------------------------------------------------------------
    def _get_alltable_names(self, filter_for: Optional[str] = None):
        sql = """SELECT table_name FROM information_schema.tables;"""
        with Mnemosyne() as conn:
            table_names = conn.query(sql)

        if table_names:
            table_names = [item[0] for item in table_names]
            if filter_for:
                return [item for item in table_names if filter_for in item]
            return table_names
        return []

    def _get_max_workers(self, interval: str):
        if self.exchange_name == "binance":
            _max_sim = 40
            seconds_per_interval = interval_to_milliseconds(interval) / 1000
            _broker = max(1, int(round(_max_sim * 500 / seconds_per_interval)))
            _hermes = max(1, int(round(_max_sim / _broker)))
            return _hermes, _broker

        else:
            return 1, 1

    def _rename_binance_tables(self):
        def _rename_single_table(old_new: tuple, conn=None):
            old_tablename = old_new[0]
            new_tablename = old_new[1]

            sql = f"""ALTER TABLE {old_tablename} RENAME {new_tablename};"""
            logger.debug(sql)
            conn.query(sql)

        names = self._get_alltable_names()
        names = [n[0] for n in names if "prices" in n[0]]
        splitted = [n.split("_") for n in names]
        new = ["_".join(["binance", item[0], "ohlcv", item[2]]) for item in splitted]
        combined = zip(names, new)
        with Mnemosyne() as conn:
            [_rename_single_table(item, conn) for item in combined]


# ===============================================================================
class Hermes(HermesDataBase):
    """Hermes is the repository for all general (public) informations
    that we might need. He knows everything about symbols and prices
    (OHLCV) and who knows what else ...

    The information is retrieved from the database if we have it there
    or from the exchange API.
    """

    def __init__(self, exchange: str, mode: str = "live", verbose: bool = False):
        """Hermes initialization ...

        :param exchange: name of the exchange, 'binance' or 'kucoin'
        :type exchange: str
        :param mode: 'live' or 'backtest' (the latter adds 200
        datapoints to OHLCV queries), defaults to 'live'
        :type mode: str, optional
        :param verbose: talk more or less during operation, defaults to False
        :type verbose: bool, optional
        """
        # Log the calling function and line number
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back
        if caller_frame:
            caller_info = inspect.getframeinfo(caller_frame)
            logger.debug(
                f"Hermes __init__ called from {caller_info.filename}, "
                f"line {caller_info.lineno}, function {caller_info.function}"
                )
        else:
            logger.debug("Hermes __init__ called, but couldn't determine caller")

        self.name = "HERMES"
        self.mode = mode
        self.verbose = verbose

        if self.mode == "backtest":
            self.updateAfter = 3600 * 24 * 180

        self.exchange_name: str = self._set_exchange(exchange)
        self.exchange: Callable = self._get_exchange_obj(exchange)

        HermesDataBase.__init__(self, self.exchange_name, self.exchange)

        logger.debug(f"{self.name} initializing in {self.mode} mode:")

        self.symbol = ""
        self.interval = ""
        self.start = None
        self.end = None
        self.lookback = 200

        self.table_is_empty = True

        self.all_symbols = {}
        self._tradeable_symbols = []
        self._initialize_symbol_cache(self.exchange_name)

        if self.verbose:
            logger.debug("OK")
            logger.debug(f"exchange:   {self.exchange_name}")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    @property
    def tradeable_symbols(self):
        return self.get_tradeable_symbols()

    # --------------------------------------------------------------------------
    def get_tradeable_symbols(
        self,
        quote_asset: str = "",
        remove_leveraged: bool = False,
        margin_only: bool = False,
    ) -> list[str]:
        """Get all tradable (active) symbols (for a specific quote asset)

        :param quote_asset: filter result for quote asset, defaults to None
        :type quote_asset: str, optional
        :param remove_leveraged: exclude (or not) all leveraged symbols,
                                 only relevant for Binance, defaults to False
        :type remove_leveraged: bool, optional
        :param margin_only: return only symbols that can be traded on margin
        :type margin_only: bool
        :raises ValueError: if quote_asset is not s string
        :return: [description]
        :rtype: list
        """
        logger.debug(f"fetching all symbols for {self.exchange_name}")
        _all = self.all_symbols.get(self.exchange_name, {})
        logger.debug(self.all_symbols.keys())

        if not all:
            sys.exit()
            return []

        # remove symbols that are not currently trading from result
        _all = {k: v for k, v in _all.items() if v["status"] != "BREAK"}

        # filter out symbols where margin trading is available if requested
        if margin_only:
            _all = {k: v for k, v in _all.items() if "MARGIN" in v["permissions"]}

        # remove the leveraged (but SPOT traded) symbols if requested
        if self.exchange_name == "binance" and remove_leveraged:
            _up = [sym for sym in _all if "UP" in sym or "BULL" in sym]
            _down = [sym for sym in _all if "DOWN" in sym or "BEAR" in sym]
            _forbidden = _up + _down
            _all = {k: v for k, v in _all.items() if k not in _forbidden}

        elif self.exchange_name == "kucoin" and remove_leveraged:
            _all = {k: v for k, v in _all.items() if "3" not in v.get("symbol")}

        if not quote_asset:
            return [sym for sym in _all.keys()]
        else:
            return [sym for sym in _all if _all[sym]["quoteAsset"] == quote_asset]

    # ============== general symbol information related methods ====================

    def get_symbol(self, symbol: str) -> dict:
        def _we_need_to_update(sym: dict):
            _updated = result.get("updateTime", 0)
            _now = time.time()
            time_since_update = time_difference_in_ms(_updated, _now) / 1000

            if time_since_update < self.updateAfter:
                return False

            return True

        # ....................................................................
        logger.debug("[HERMES] fetching symbol from cache")
        result = self._get_symbol_from_cache(symbol)

        if result:
            logger.debug(f"found {symbol} in cache ...")
            if not _we_need_to_update(result):
                return result
            else:
                logger.debug("updating symbols from database ...")
                self._update_symbols_table(get_exchange_name(symbol))
                result = self._get_symbol_from_database(symbol=symbol)
                if result:
                    return result
        else:
            logger.debug("updating symbols from API ...")
            res = self._get_symbol_from_exchange(symbol)
            try:
                symbol = self._flatten_symbol_dictionary(res)
                return self._convert_symbol_values(symbol)
            except Exception:
                result = self._get_symbol_from_database(symbol=symbol)
                if result:
                    return result
                else:
                    raise Exception(f"unable to get symbol information for {symbol}")

    # ..........................................................................
    def _get_symbol_from_exchange(self, symbol: str) -> dict:
        self.set_symbol(symbol)
        with self.exchange() as conn:
            return conn.get_symbol(symbol)

    def _get_symbol_from_database(self, symbol: str) -> dict:
        with Mnemosyne() as conn:
            symbol = conn.get_symbol(symbol)
            if symbol:
                return self._convert_symbol_values(symbol)
        return None

    def _get_symbol_from_cache(self, symbol: str) -> dict:
        return self.all_symbols[self.exchange_name].get(symbol)

    def _get_all_symbols_from_exchange(self, exchange_name: str) -> dict:
        exchange = self._get_exchange_obj(exchange_name)
        with exchange() as conn:
            res = conn.get_symbols()

        if "message" in res:
            res = res["message"]

        all_symbols = [self._flatten_symbol_dictionary(sym) for sym in res]
        return [self._convert_symbol_values(sym) for sym in all_symbols]

        return None

    def _get_all_symbols_from_database(self, exchange_name: str) -> list:
        with Mnemosyne() as conn:
            try:
                all_symbols = conn.get_all_symbols_information(exchange_name)
            except Exception as e:
                logger.debug(f"{e} for {exchange_name.upper()}")
                return []

        if all_symbols:
            return [self._convert_symbol_values(s) for s in all_symbols]

    def _flatten_symbol_dictionary(self, data: dict) -> dict:
        # helper function to extract the nested 'filters' dictioanry
        def _get_filters(_data: dict, filters_raw: dict) -> dict:
            for _filter in filters_raw:
                t = _filter.get("filterType").split("_")
                type = t[0].lower() + "".join(
                    [word.capitalize() for idx, word in enumerate(t) if idx > 0]
                )
                for k, v in _filter.items():
                    if not k == "filterType":
                        if type == "maxNumAlgoOrders" or type == "maxNumOrders":
                            key = f"f_{type}"
                        else:
                            key = "f_" + "_".join([type, k])

                        _data[key] = v
            return _data

        # ......................................................................
        _flattened = {}

        # flatten the original (nested) dictionary ('filters' is
        # the nested dict) and convert lists to comma separated strings
        try:
            for k, v in sorted(data.items()):
                if not k == "filters":
                    if isinstance(v, list):
                        v = ",".join(v)
                    _flattened[k] = v
                else:
                    _flattened = _get_filters(_flattened, v)
        except Exception as e:
            logger.debug(e)
            logger.debug(data)
            sys.exit()

        # add an update time field for later use
        _flattened["updateTime"] = int(time.time())

        # for a few symbols (leveraged ones for instance) the Binance
        # format is has more or less values than all the others and
        # we need to standardize these before writing to the database
        if "f_marketLotSize_stepSize" not in _flattened.keys():
            _flattened["f_marketLotSize_minQty"] = _flattened["f_lotSize_minQty"]
            _flattened["f_marketLotSize_maxQty"] = _flattened["f_lotSize_maxQty"]
            _flattened["f_marketLotSize_stepSize"] = _flattened["f_lotSize_stepSize"]

        valid_keys = [
            "baseAsset",
            "baseAssetPrecision",
            "baseCommissionPrecision",
            "cancelReplaceAllowed",
            "f_icebergParts_limit",
            "f_lotSize_maxQty",
            "f_lotSize_minQty",
            "f_lotSize_stepSize",
            "f_marketLotSize_maxQty",
            "f_marketLotSize_minQty",
            "f_marketLotSize_stepSize",
            "f_maxNumOrders",
            "f_maxNumAlgoOrders",
            "f_minNotional_applyToMarket",
            "f_minNotional_avgPriceMins",
            "f_minNotional_minNotional",
            "f_percentPrice_avgPriceMins",
            "f_percentPrice_multiplierDown",
            "f_percentPrice_multiplierUp",
            "f_priceFilter_maxPrice",
            "f_priceFilter_minPrice",
            "f_priceFilter_tickSize",
            "f_trailingDelta_maxTrailingAboveDelta",
            "f_trailingDelta_maxTrailingBelowDelta",
            "f_trailingDelta_minTrailingAboveDelta",
            "f_trailingDelta_minTrailingBelowDelta",
            "icebergAllowed",
            "isSpotTradingAllowed",
            "isMarginTradingAllowed",
            "ocoAllowed",
            "orderTypes",
            "permissions",
            "quoteAsset",
            "quoteAssetPrecision",
            "quoteCommissionPrecision",
            "quoteOrderQtyMarketAllowed",
            "quotePrecision",
            "status",
            "symbol",
            "updateTime",
        ]

        return {k: v for k, v in sorted(_flattened.items()) if k in valid_keys}

    def _convert_symbol_values(self, data: dict) -> dict:
        _floats = [
            "f_lotSize_maxQty",
            "f_lotSize_minQty",
            "f_lotSize_stepSize",
            "f_marketLotSize_maxQty",
            "f_marketLotSize_minQty",
            "f_marketLotSize_stepSize",
            "f_minNotional_minNotional",
            "f_percentPrice_avgPriceMins",
            "f_percentPrice_multiplierDown",
            "f_percentPrice_multiplierUp",
            "f_priceFilter_maxPrice",
            "f_priceFilter_minPrice",
            "f_priceFilter_tickSize",
        ]

        _ints = [
            "baseAssetPrecision",
            "baseCommissionPrecision",
            "f_icebergParts_limit",
            "f_maxNumAlgoOrders",
            "f_maxNumOrders",
            "f_minNotional_avgPriceMins",
            "quoteAssetPrecision",
            "quoteCommissionPrecision",
            "quotePrecision",
        ]

        _lists = ["orderTypes", "permissions"]

        _bools = [
            "f_minNotional_applyToMarket",
            "icebergAllowed",
            "isMarginTradingAllowed",
            "isSpotTradingAllowed",
            "ocoAllowed",
            "quoteOrderQtyMarketAllowed",
        ]
        try:
            data = {
                k: float(v) if (k in _floats) and (v != "") else v
                for k, v in data.items()
            }
            data = {
                k: int(v) if (k in _ints) and (v != "") else v for k, v in data.items()
            }
            data = {k: v.split(",") if k in _lists else v for k, v in data.items()}
            data = {
                k: (True if v == 1 else False) if k in _bools else v
                for k, v in data.items()
            }
        except Exception as e:
            logger.debug(e)
            pprint(data)

        return data

    def _initialize_symbol_cache(self, exchange_name: str):
        logger.debug("initializing symbol cache ...")

        _res = self._get_all_symbols_from_database(exchange_name)

        if _res:
            logger.debug(f"loaded symbols from database for {exchange_name}")
            self.all_symbols[exchange_name] = {v["symbol"]: v for v in _res}
        else:
            logger.debug(
                f"failed to get symbols from database! for {self.exchange_name}"
            )
            logger.debug(_res)
            self.all_symbols[exchange_name] = {}

    # ========================= ohlcv related methods ================================
    @lru_cache
    def get_ohlcv(
        self,
        symbols: Union[str, list, tuple],
        interval: str,
        start: Union[int, str],
        end: Union[int, str],
    ) -> dict:
        """Get the prices (ohlcv data) for a symbol or a list of symbols

        :param symbols: name of symbol or list with names of symbols
        :type symbols: str|list
        :param interval: interval for ohlcv (1m, 3m, ... 12, 1d, 1w)
        :type interval: str
        :param start: start of time period - date or unix timestamp (UTC)
        :type start: int|str
        :param end: end of time period - date or unix timestamp (UTC)
        :type end: int|str
        """
        # check if we got one symbol or a list with multiple symbols
        symbols = [symbols] if isinstance(symbols, str) else symbols

        self._set_exchange_from_symbol(symbols[0])
        start, end = self.get_timestamps(start, end, interval)

        self.interval = interval
        self.start, self.end = start, end

        logger.info(
            f"fetching data for {symbols} on {self.exchange_name.upper()} "
            f"for period: {unix_to_utc(start)} - {unix_to_utc(end)}"
        )

        if len(symbols) == 1:
            return self._get_ohlcv_for_one_symbol(
                symbol=symbols[0], interval=interval, start=start, end=end
            )

        # ......................................................................
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures = []
            for sym in symbols:
                kwargs = {
                    "symbol": sym,
                    "interval": interval,
                    "start": start,
                    "end": end,
                }

                futures.append(
                    executor.submit(self._get_ohlcv_for_one_symbol, **kwargs)
                )

            futures, _ = concurrent.futures.wait(futures)
            for future in futures:
                results.append(future.result())

        self.start, self.end, self.interval = None, None, None

        if len(results) > 1:
            return results
        else:
            return results[0]

    # ................................................................................
    def _get_ohlcv_for_one_symbol(
        self, symbol: str, interval: str, start: int, end: int
    ) -> dict:
        """Gets OHLCV data for one symbol."""

        if self.exchange is None:
            self._set_exchange_from_symbol(symbol)

        interval_in_ms = interval_to_milliseconds(interval)

        if interval_in_ms is None:
            raise ValueError(f"Bad interval: {interval}")

        # list of intervals that are not in our database (bc: too much data)
        if interval in ["1m", "3m", "5m"]:
            res = self._get_ohlcv_from_api(
                symbol=symbol, interval=interval, start=self.start, end=self.end
            )

            res["symbol"], res["interval"] = symbol, interval
            return res

        # fetch the data from the database
        try:
            res = self._get_ohlcv_from_database(
                symbol=symbol, interval=interval, start=start, end=end
            )
        except BadSymbolError:
            return {"success": False, "error": "Bad Symbol"}
        except Exception as e:
            logger.debug(f"error while trying to retrieve data from database: {e}")
            return {"success": False, "error": str(e)}

        if res is not None:
            return {
                "success": True,
                "message": res,
                "symbol": symbol,
                "interval": interval,
                "error": None,
            }
        else:
            logger.error(
                "unknown error while trying to retrieve data from database",
                )
            return {
                "symbol": symbol,
                "success": False,
                "error": "No data found for the given time period",
            }

    def _get_ohlcv_from_api(
        self, symbol: str, interval: str, start: int, end: int
    ) -> pd.DataFrame:
        """Gets OHLCV data from exchange API"""

        end = int(end / 1000)

        with self.exchange() as conn:
            return conn.get_ohlcv(
                symbol=symbol, interval=interval, start=start, end=end
                )

    def _standardize_kucoin_kline(self, kline: list, interval: str) -> list:
        """Transforms raw Kucoin OHLCV data to standard format.

        :param kline: kline in Kucoin format
        :type kline: list
        :raises TypeError: [description]
        :return: kline as list in Binance format
        :rtype: list

        This is what we get from Kucoin:
        .. code:: python
            [
                "1545904980",             //Start time of the candle cycle
                "0.058",                  //opening price
                "0.049",                  //closing price
                "0.058",                  //highest price
                "0.049",                  //lowest price
                "0.018",                  //Transaction volume
                "0.000945"                //Transaction amount
            ]
        """
        try:
            _open_time = int(kline[0]) * 1000  # is seconds, but we want milliseconds
            _open = kline[1]
            _high = kline[3]
            _low = kline[4]
            _close = kline[2]
            _close_time = _open_time + interval_to_milliseconds(interval) - 1
            _volume = kline[5]
            _quote_volume = kline[6]

            return [
                _open_time,
                _open,
                _high,
                _low,
                _close,
                _volume,
                _close_time,
                _quote_volume,
                "0",
                "0",
                "0",
                "0",
            ]

        except ValueError as e:
            logger.debug(e)
            logger.debug(kline[0])

        except IndexError as e:
            logger.debug(e)
            pprint(kline)

    # ============================= helper methods =================================
    def _set_exchange(self, exchange_name: str) -> str:
        if exchange_name.lower() in VALID_EXCHANGES:
            self.exchange_name = exchange_name.lower()
            self.exchange = VALID_EXCHANGES[self.exchange_name].get("api_class")

            if self.exchange_name == "kucoin":
                self._max_workers = MAX_WORKERS_KUCOIN
            else:
                self._max_workers = MAX_WORKERS_BINANCE

            return exchange_name.lower()
        else:
            raise ValueError("unknown exchange: {exchange_name}")

    def _get_exchange_from_symbol(self, symbol: str) -> object:
        """Determine exchange based on symbol name.

        :param symbol: name of the symbol
        :type symbol: str
        :raises TypeError: TypeError if symbol is not a string
        :return: exchange class object
        :rtype: object
        """
        if isinstance(symbol, str):
            if "-" in symbol:
                return Kucoin
            else:
                return Binance
        else:
            raise TypeError(symbol)

    def _set_exchange_from_symbol(self, symbol: str):
        exchange_name = "kucoin" if "-" in symbol else "binance"
        self._set_exchange(exchange_name)

    def set_symbol(self, symbol):
        self.symbol = symbol
        self._set_exchange_from_symbol(symbol)

    def set_interval(self, interval):
        self.interval = interval

    def get_timestamps(
        self, start: Union[str, int], end: Union[str, int], interval: str
    ) -> Tuple[int, int]:
        start, end = get_start_and_end_timestamp(
            start=start, end=end, interval=interval, verbose=self.verbose
        )

        if self.mode == "backtest":
            start = self._add_buffer_to_start(start, interval)

        return int(start), int(end)

    def _get_table_name(self):
        if self.exchange_name == "kucoin":
            symbol_name = "".join(self.symbol.split("-"))

        return "_".join([self.exchange_name, symbol_name, "ohlcv", self.interval])

    def get_listing_date(self, symbol: str, interval: str = "1d") -> int:
        res = self.exchange()._get_earliest_valid_timestamp(
            symbol=symbol, interval=interval
        )

        if isinstance(res, int):
            return res
        elif isinstance(res, str):
            return int(res)
        elif isinstance(res, dict):
            if "success" in res and res["success"]:
                return int(res["message"])

        raise Exception(f"unable to get listing date for {symbol} ({interval})")

    # ------------------------------------------------------------------------
    # helper functions
    def _add_buffer_to_start(self, start, interval):
        """Add a buffer to the start of the period.

        :param start: start time of the period
        :type start: int
        :param interval: interval of the period
        :type interval: str
        :return: start time of the period with buffer
        :rtype: int
        :raises TypeError: TypeError if interval is not a string
        :raises ValueError: ValueError if interval is not a valid interval
        """
        if intrvl_in_ms := interval_to_milliseconds(interval):
            return start - intrvl_in_ms * self.lookback
        else:
            raise ValueError("Invalid interval: {interval}")

    def _get_exchange_obj(self, exchange_name: str) -> Callable:
        """Get an instance of the exchange class.

        :param exchange_name: name of the exchange
        :type exchange_name: str

        :returns: exchange object (Binance/Kucoin)
        :rtype: instance of the exchange class

        :raises: ValueError
        """
        if not exchange_name.lower() in VALID_EXCHANGES:
            valid_exchanges = tuple(VALID_EXCHANGES.keys())
            raise ValueError(
                f"Exchange must be one of {valid_exchanges} ({exchange_name})"
            )
        else:
            exc_obj = VALID_EXCHANGES.get(exchange_name.lower(), {}).get("api_class")

            if exc_obj is None:
                raise ValueError(f"unable to set API class for {exchange_name}")

            return exc_obj
