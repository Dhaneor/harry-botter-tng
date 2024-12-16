#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:21:53 2021

@author: dhaneor
"""
import logging
from time import time
from typing import Union, Optional

import mysql.connector as dbc
from mysql.connector import Error, IntegrityError

from src.helpers.ilabrat import get_exchange_name
from src.util.timeops import utc_to_unix

import pandas as pd
from pprint import pprint

logger = logging.getLogger("main.mnemosyne")
logger.setLevel(logging.INFO)

# ============================================================================


class Mnemosyne:
    def __init__(self):
        self.name = "Mnemosyne"
        # print(f'{self.name} initializing:', end=' ')

        self.description = "Keeper of the Archives"
        self.db_ip = "127.0.0.1"
        # self.db_ip = '85.214.71.96'
        self.db_name = "akasha"
        self.db_user = self.name
        self.db_pass = "cp!8R//G6EHhTaO3F6wrx"
        self.auth_plugin = "mysql_native_password"

        self.error = []
        self.verbose = False

        self.conn = None
        self.connect()

        self.start = 0
        self.end = time()

    def __enter__(self):
        self.connect()
        logger.debug(f"{self.name} entering - connected to {self.db_name}")
        return self

    def __exit__(self, *args, **kwargs):
        self.disconnect()
        logger.debug(f"{self.name} exiting - disconnected from {self.db_name}")
        return False

    # ============================================================================

    def connect(self):
        if self.db_ip != "127.0.0.1":
            port = 33006
        else:
            port = 3306

        try:
            self.conn = dbc.connect(
                host=self.db_ip,
                port=port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_pass,
                auth_plugin=self.auth_plugin,
            )

            if self.conn.is_connected():
                return True

        except Error as e:
            print(e)
            # if 'Access denied' in e.msg:
            #     raise ConnectionRefusedError(e.msg)
            # else:
            #     raise ConnectionError(e)

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def query(self, sql):
        if self.conn is None:
            if not self.connect():
                return

        if self.conn is None:
            raise ConnectionError(f"Connection to {self.db_name} failed")

        try:
            logger.debug(f"QUERY for {sql} -> start")
            cursor = self.conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
        except Error as e:
            logger.error(f"QUERY FAIL for {sql} -> error{e} ")
            result = []
        except Exception as e:
            logger.error(f"QUERY FAIL for {sql} -> error{e} ")
            result = []
        finally:
            cursor.close()

        return result

    # ============================================================================
    def create_table(self, table_description: str):
        self.query(table_description)

    def drop_table(self, table_name):
        sql = f"DROP TABLE IF EXISTS {table_name}"
        self.query(sql)

    def drop_multiple_tables(self, list_of_tables):
        for table in list_of_tables:
            print(f"Deleting table {table}:", end=" ")
            if self.drop_table(table):
                print("OK")
        else:
            print("FAIL")

    # -------------------------------------------------------------------------
    def get_all_table_names(self, containing: Optional[str] = None):
        if containing is None:
            sql = "SHOW tables"
        else:
            sql = f"SHOW tables LIKE '%{containing}%'"

        tables = self.query(sql)
        if tables:
            return [item[0] for item in tables]

        return []

    def get_ohlcv_table_status(self, table_name) -> dict:
        no_of_rows, earliest, latest = 0, 0, 0
        exists = self.check_if_table_exists(table_name)

        if exists:
            sql = f"SELECT count(*) FROM {table_name}"
            no_of_rows = self.query(sql)
            if no_of_rows:
                no_of_rows = int(no_of_rows[0][0])
            else:
                return {}

            if no_of_rows > 0:
                sql = f"SELECT MAX(openTime) from {table_name}"
                latest = self.query(sql)
                latest = int(latest[0][0]) if latest else 0

                sql = f"SELECT MIN(openTime) from {table_name}"
                earliest = self.query(sql)
                earliest = int(earliest[0][0]) if earliest else 0

        return {
            "name": table_name,
            "table exists": exists,
            "number of entries": no_of_rows,
            "earliest open": earliest,
            "latest open": latest,
        }

    def check_if_table_exists(self, table_name: str) -> bool:
        sql = f"SHOW TABLES LIKE '{table_name}'"
        result = self.query(sql)

        if result and len(result) != 0:
            return True

        return False

    def get_number_of_entries_in_table(self, table_name):
        if self.check_if_table_exists(table_name):
            sql = f"SELECT count(*) FROM {table_name}"
            result = self.query(sql)

            return result[0][0]

        else:
            return -1

    def get_column_names_from_table(self, table_name: str) -> list:
        sql = f"""select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS \
                    where TABLE_NAME='{table_name}'"""
        columns = self.query(sql)
        if columns:
            return [col[0] for col in columns]

    def get_latest_timestamps(
        self, exchange: str, symbols: list, intervals: list, all_tables=None
    ) -> None:
        if not all_tables:
            all_tables = self.get_all_table_names()

        print(f"Checking the timestamps for {len(all_tables)} tables:", end=" ")
        result = []

        need_update = 0
        no_entries = 0
        no_table = 0

        # print(f'Got {len(symbols)} symbols and {len(intervals)} intervals ...')

        # check if a table exists for symbol and interval
        # if yes ... get the latest timestamp or use one from before binance launch
        # if no ... return False so the calling functions knows to create one
        # create and return a list with the results
        for symbol in symbols:
            for interval in intervals:
                table_name = self._get_ohlcv_table_name(symbol, interval)
                item = []

                if table_name in all_tables:
                    if self.get_number_of_entries_in_table(table_name) > 0:
                        sql = f"SELECT MAX(openTime) from {table_name}"
                        last_ts = self.query(sql)
                        last_ts = int(last_ts[0][0])

                        item = [symbol, interval, table_name, True, last_ts]
                        need_update += 1

                    else:
                        item = [symbol, interval, table_name, True, 1498860000]
                        no_entries += 1
                        need_update += 1

                else:
                    item = [symbol, interval, table_name, False, 1498860000]
                    no_table += 1

                result.append(item)
        print("OK")
        print(f"{need_update} tables in our database have a valid timestamp")
        print(f"{no_entries} tables are empty")
        print(f"{no_table} tables need to be created")
        print("•" * 60)

        return result

    # ------------------------------------------------------------------------
    def save_to_database(
        self, table_name: str, columns: list, data: list, batch=False
    ) -> bool:
        """This method is for saving entries in an existing table.

        :param table_name: name of the table
        :type table_name: str
        :param columns: a list of the columns to be used
        :type columns: str
        :param data: the data that needs to be saved
        :type data: tuple or list of tuples
        """
        if not self.conn:
            print("No database connection ...")
            self.connect()

        cursor = self.conn.cursor()
        len_data = len(data)

        try:
            _columns = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            sql = (
                f"INSERT IGNORE INTO {table_name} ({_columns}) VALUES ({placeholders})"
            )

            if batch:
                logger.debug(f"BATCH INSERT for {table_name} with {len_data} entries")
                cursor.executemany(sql, data)
            else:
                logger.debug(f"INSERT for {table_name} with {len_data} entries")
                cursor.execute(sql, data)

            self.conn.commit()

        except IntegrityError as err:
            logger.debug(f"Integrity Error for {table_name} -> {err}")
            return False
        except Exception as e:
            logger.error(f"INSERT FAIL for {table_name} -> error{e} ")
        finally:
            cursor.close()

        return True

    def update(
        self,
        table_name,
        columns,
        data,
    ):
        pass
        # REPLACE into table (id, name, age) values(1, "A", 19)

    # ========================================================================
    # functions for special queries
    def _create_symbols_table(self):
        table_name = "symbols"

        table_description = (
            """
                        CREATE TABLE IF NOT EXISTS `%s` (
                        id INTEGER PRIMARY KEY AUTO_INCREMENT,
                        symbol VARCHAR(15) NOT NULL UNIQUE,
                        status VARCHAR(15) NOT NULL,
                        baseAsset VARCHAR(10) NOT NULL,
                        baseAssetPrecision INT NOT NULL,
                        quoteAsset VARCHAR(10) NOT NULL,
                        quotePrecision INT NOT NULL,
                        quoteAssetPrecision INT NOT NULL,
                        baseCommissionPrecision INT NOT NULL,
                        quoteCommissionPrecision INT NOT NULL,
                        orderTypes VARCHAR(127) NOT NULL,
                        icebergAllowed TINYINT NOT NULL,
                        ocoAllowed TINYINT NOT NULL,
                        quoteOrderQtyMarketAllowed TINYINT NOT NULL,
                        isSpotTradingAllowed TINYINT NOT NULL,
                        isMarginTradingAllowed TINYINT NOT NULL,
                        permissions VARCHAR(127) NOT NULL,
                        f_priceFilter_maxPrice VARCHAR(31),
                        f_priceFilter_minPrice VARCHAR(31),
                        f_priceFilter_tickSize VARCHAR(31),
                        f_percentPrice_avgPriceMins VARCHAR(31),
                        f_percentPrice_multiplierDown VARCHAR(31),
                        f_percentPrice_multiplierUp VARCHAR(31),
                        f_lotSize_maxQty VARCHAR(31),
                        f_lotSize_minQty VARCHAR(31),
                        f_lotSize_stepSize VARCHAR(31),
                        f_minNotional_applyToMarket VARCHAR(31),
                        f_minNotional_avgPriceMins VARCHAR(31),
                        f_minNotional_minNotional VARCHAR(31),
                        f_icebergParts_limit TINYINT(31),
                        f_marketLotSize_maxQty VARCHAR(31),
                        f_marketLotSize_minQty VARCHAR(31),
                        f_marketLotSize_stepSize VARCHAR(31),
                        f_MaxNumOrders INT,
                        f_MaxNumAlgoOrders INT
                        ) """
            % table_name
        )

        self.create_table(table_name, table_description)

        return

    def get_list_of_all_symbols(
        self, exchange_name: str, return_dataframe: bool = False
    ) -> Union[list, pd.DataFrame]:
        table_name = f"{exchange_name.lower()}_symbols"
        sql = f"SELECT symbol FROM {table_name};"
        result = self.query(sql)

        if return_dataframe:
            return self.symbols_to_dataframe(result)
        else:
            return result

    def get_all_symbols_information(self, exchange_name: str) -> list:
        table_name = f"{exchange_name.lower()}_symbols"
        sql = f"SELECT * FROM {table_name};"
        all_symbols = self.query(sql)

        if all_symbols:
            columns = self.get_column_names_from_table(table_name)
            return [dict(zip(list(columns), list(s))) for s in all_symbols]
        else:
            print(f"[MENOMOSYNE] could not find symbols for {exchange_name}")
            print(sql)
            print(all_symbols)
            print(self.get_all_table_names(containing="symbols"))
            print(self.get_column_names_from_table(table_name))
            return []

    def get_symbol(self, symbol: str):
        table_name = f"{get_exchange_name(symbol)}_symbols"

        sql = f"""SELECT * FROM {table_name} WHERE symbol = '{symbol}';"""
        values = self.query(sql)

        if values:
            values = list(values[0])
            columns = self.get_column_names_from_table(table_name)
            if columns:
                return {k: v for k, v in (zip(columns, values))}

    # -------------------------------------------------------------------------
    # get the ohclv data in the same format as a direct
    # binance api request returns (= dataframe)
    def get_ohclv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start: int,
        end: int,
        return_as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, dict, None]:
        if exchange == "kucoin":
            symbol = "".join(symbol.split("-"))

        table_name = "_".join([exchange, symbol, "ohlcv", interval])

        sql = (
            f"SELECT * FROM {table_name} "
            f"WHERE openTime BETWEEN {start} AND {end} "
            "ORDER BY openTime;"
        )

        result = self.query(sql)

        if not result:
            return None

        if return_as_dataframe:
            return self.ohlcv_db_result_to_dataframe(result)
        else:
            return result

    # get a list of all binance symbols/markets for a given quote asset
    def get_symbols_for_quote_asset(self, quote_asset):
        symbols_table = "symbols"

        sql = (
            f"""SELECT symbol FROM {symbols_table} WHERE quoteAsset = '{quote_asset}'"""
        )
        result = self.query(sql)

        symbols = []
        for i in range(len(result)):
            symbol = result[i][0]
            symbols.append(symbol)

        return symbols

    # create the price table for a given symbol and interval
    def create_price_table_for_symbol(self, symbol, interval):
        table_name = symbol + "_prices_" + interval

        # prepare sql statement create table
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

        self.create_table(table_name, table_description)
        return True

    def _get_ohlcv_table_name(self, symbol: str, interval: str):
        symbol_name = "".join(symbol.split("-")) if "-" in symbol else symbol
        return "_".join([get_exchange_name(symbol), symbol_name, "ohlcv", interval])

    # ========================================================================
    # helper methods
    def set_start_time(self, start_time):
        self.start = utc_to_unix(start_time)
        print(f"{self.name}: start time = {self.start} - {start_time}")

    def set_end_time(self, end_time):
        self.end = utc_to_unix(end_time)
        print(f"{self.name}: end time = {self.end} - {end_time}")

    def ohlcv_db_result_to_dataframe(self, result):
        df = pd.DataFrame(
            result,
            columns=[
                "Index",
                "open time",
                "human open time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close time",
                "quote asset volume",
                "number of trades",
                "taker buy base asset volume",
                "taker buy quote asset volume",
            ],
        )

        # change the data types
        df["open time"] = df["open time"].astype(int)
        df["human open time"] = pd.to_datetime(df["human open time"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["close time"] = df["close time"].astype(str)
        df["quote asset volume"] = df["quote asset volume"].astype(float)
        df["taker buy base asset volume"] = df["taker buy base asset volume"].astype(
            float
        )
        df["taker buy quote asset volume"] = df["taker buy quote asset volume"].astype(
            float
        )

        drop_columns = [
            "Index",
            "number of trades",
            "taker buy base asset volume",
            "taker buy quote asset volume",
        ]
        df.drop(columns=drop_columns, axis=1, inplace=True)

        return df

    def symbols_to_dataframe(self, symbols_from_db):
        df = pd.DataFrame(
            symbols_from_db,
            columns=[
                "id",
                "symbol",
                "status",
                "baseAsset",
                "baseAssetPrecision",
                "quoteAsset",
                "quotePrecision",
                "quoteAssetPrecision",
                "baseCommissionPrecision",
                "quoteCommissionPrecision",
                "orderTypes",
                "icebergAllowed",
                "ocoAllowed",
                "quoteOrderQtyMarketAllowed",
                "isSpotTradingAllowed",
                "isMarginTradingAllowed",
                "permissions",
                "f_priceFilter_maxPrice",
                "f_priceFilter_minPrice",
                "f_priceFilter_tickSize",
                "f_percentPrice_avgPriceMins",
                "f_percentPrice_multiplierDown",
                "f_percentPrice_multiplierUp",
                "f_lotSize_maxQty",
                "f_lotSize_minQty",
                "f_lotSize_stepSize",
                "f_minNotional_applyToMarket",
                "f_minNotional_avgPriceMins",
                "f_minNotional_minNotional",
                "f_icebergParts_limit",
                "f_marketLotSize_maxQty",
                "f_marketLotSize_minQty",
                "f_marketLotSize_stepSize",
                "f_maxNumOrders",
                "f_maxNumAlgoOrders",
            ],
        )

        return df

    def talk_to_me(self, yesno: bool):
        if yesno:
            self.verbose = True
        else:
            self.verbose = False

        return

    # ------------------------------------------------------------------------
    """
    # COMMENTED OUT, BECAUSE ... YOU SHOULD NEVER DO THIS, RIGHT?

    def drop_all_tables():

        tables = self.get_all_table_names()

        if len(tables) != 0:

            self.drop_multiple_tables(tables)


            tables = mnemosyne.get_all_table_names()

            if len(tables)  == 0:

                print('Done!')

        else:

            print(f'There are no tables in our database!')


        return
    """


# =============================================================================
if __name__ == "__main__":
    symbol = "QNTBTC"
    interval = "1d"

    with Mnemosyne() as conn:
        x = conn.get_all_table_names(containing="kucoin")

    if x:
        pprint(x)
    else:
        print("MEH")

    # _all = m.get_all_table_names()
    # ku = [table for table in _all if '<' in table]
    # print(ku)
    # res = m.drop_table(ku[0])
