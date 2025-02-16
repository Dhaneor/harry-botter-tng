#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:11:33 2021

@author dhaneor
"""

import sys
import os
import pandas as pd
import logging
from pprint import pprint
from typing import Union, Optional
from random import choice

LOG_LEVEL = "INFO"
logger = logging.getLogger("main")
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)


# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from src.staff.hermes import Hermes  # noqa: E402, F401
from src.util.timeops import unix_to_utc, execution_time  # noqa: E402


# =============================================================================
class HermesTester:

    def __init__(self, exchange: str, interval: str, verbose: bool = False):
        self.exchange = exchange
        self.interval = interval
        self.hermes = Hermes(exchange=exchange, mode="live", verbose=verbose)

    # -------------------------------------------------------------------------
    def set_symbol(self, symbol_name: str):
        self.symbol_name = symbol_name

    def set_interval(self, interval: str):
        self.interval = interval

    # -------------------------------------------------------------------------
    @execution_time
    def create_symbols_table(self):
        self.hermes.set_symbol(self.symbol_name)
        self.hermes._create_symbols_table(self.hermes.exchange_name)

    def drop_symbols_table(self):
        self.hermes._drop_symbols_table("kucoin_symbols")

    @execution_time
    def update_symbols_table(self, exchange_name: str):
        self.hermes._update_symbols_table(exchange_name)

    @execution_time
    def get_all_symbols_from_exchange(self, exchange_name: str):
        res = self.hermes._get_all_symbols_from_exchange(exchange_name)
        if res:
            pprint(res[0])

        print(f"found {len(res)} symbols for {exchange_name}")

    @execution_time
    def get_all_symbols_from_database(self, exchange_name: str):
        res = self.hermes._get_all_symbols_from_database(exchange_name)
        if res:
            pprint(res[0])

    # -------------------------------------------------------------------------
    @execution_time
    def get_symbol(self, symbol_name: str):

        symbol = self.hermes.get_symbol(symbol=symbol_name)
        pprint(symbol)

    @execution_time
    def get_multiple_symbols(self, exchange_name: str):
        all_symbols = list(self.hermes.all_symbols[exchange_name].keys())
        symbols = [choice(all_symbols) for _ in range(1000)]
        for _ in symbols:
            self.get_symbol(_)

    @execution_time
    def get_tradeable_symbols(
        self, quote_asset: Optional[str] = None, margin_only: bool = False
    ):
        symbols = self.hermes.get_tradeable_symbols(
            quote_asset=quote_asset, margin_only=margin_only
        )
        # pprint(symbols)
        print(f"found {len(symbols)} symbols for quote asset {quote_asset}")

    # -------------------------------------------------------------------------
    @execution_time
    def get_ohlcv_single(self, symbol: str, interval: str, start: int, end: int):
        res = self.hermes.get_ohlcv(symbol, interval, start, end)

        if res["success"]:
            pprint(res["message"])
            if res.get("warning") is not None:
                print(res.get("warning"))
        else:
            symbol, error = res["symbol"], res["error"]
            warning = res.get("warning")
            print(f"{symbol}: {error} ({warning})")

    @execution_time
    def get_ohlcv_multiple(
        self, symbols: list, start: Union[str, int], end: Union[str, int]
    ):

        quote_asset = "USDT"

        if not isinstance(symbols, list):
            all_symbols = self.hermes.get_tradeable_symbols(quote_asset)
            symbols = [choice(all_symbols) for _ in range(30)]
            symbols = list(set(symbols))
            print("all symbols:")
            print(all_symbols)
            print(f"found {len(all_symbols)} symbols for quote asset {quote_asset}")
            print("-" * 160)
        else:
            all_symbols = symbols

        print("will get OHLCV for:")
        print(symbols)

        res = self.hermes.get_ohlcv(symbols, self.interval, start, end)

        if isinstance(res, dict):
            if res["success"]:
                pprint(res["message"])
                if res.get("warning") is not None:
                    print(res.get("warning"))
            else:
                symbol, error = res["symbol"], res["error"]
                warning = res.get("warning")
                print(f"{symbol}: {error} ({warning})")

        else:
            successful = 0
            for item in res:
                if item["success"]:
                    successful += 1
                    df = item["message"]
                    # print('-'*150)
                    if isinstance(df, pd.DataFrame):
                        _s, _i = item["symbol"], item["interval"]
                        print(f"Got {len(df)} rows for {_s} ({_i})")
                        # print(df.tail(3))
                    if item.get("warning") is not None:
                        print(item.get("warning"))

                else:
                    symbol, error = item["symbol"], item["error"]
                    warning = item.get("warning")
                    print(f"{symbol} {error} ({warning})")
                print("-" * 160)

            print(f"got data for {successful} symbols")

    # -------------------------------------------------------------------------
    def test_set_period(self, start, end):
        self.hermes.set_interval(self.interval)
        self.hermes.set_period(start=start, end=end)
        print("*" * 80)
        print(self.hermes.start)
        print(self.hermes.end)
        hours = int((self.hermes.end - self.hermes.start) / 3600000)
        print(f"length: {hours}h")

    # -------------------------------------------------------------------------
    @execution_time
    def update_ohlcv_table(self, symbol: str):
        self.hermes._update_ohlcv_table(symbol=symbol, interval=self.interval)

    @execution_time
    def get_ohlcv_table_status(self, symbol: str):
        pprint(
            self.hermes._get_ohlcv_table_status_for_symbol(
                symbol=symbol, interval=self.interval
            )
        )

    @execution_time
    def get_ohlcv_tablename(self, symbol: str, interval: str):
        name = self.hermes._get_ohlcv_tablename(symbol=symbol, interval=interval)
        print(name)

    @execution_time
    def get_all_tablenames(self, _filter: Optional[str] = None):
        pprint(self.hermes._get_all_tablenames(filter_for=_filter))

    @execution_time
    def check_if_price_table_exists(self, symbol: str, interval: str):
        res = self.hermes._check_if_price_table_exists(symbol, interval)
        print(f"table for {symbol} ({interval}) exists: {res}")

    @execution_time
    def get_symbol_and_interval_from(self):
        tn = "kucoin_BTCUSDT_ohlcv_12h"
        print(self.hermes._get_symbol_and_interval_from(tn))

    def get_max_workers(self):
        intervals = ["15m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

        for interval in intervals:
            hermes_workers, broker_workers = self.hermes._get_max_workers(interval)
            print(f"{interval}: {hermes_workers} - {broker_workers}")

    @execution_time
    def get_listing_date(self, symbol: str, interval: str = "1d"):
        ld = self.hermes.get_listing_date(symbol, interval)
        print(f"listing date for {symbol} ({interval}): {ld} ({unix_to_utc(ld)})")

    # ..........................................................................
    @execution_time
    def rename_binance_tables(self):
        self.hermes._rename_binance_tables()

    @execution_time
    def get_leveraged_symbols(self):
        all_symbols = self.hermes.all_symbols["binance"]

        _up = {k: v for k, v in all_symbols.items() if "BULL" in k}
        _down = {k: v for k, v in all_symbols.items() if "BEAR" in k}

        _forbidden = _up.update(_down)
        pprint(_up)
        sys.exit()
        all_symbols = [sym for sym in all_symbols if sym not in _forbidden]

        print(all_symbols)

    # -------------------------------------------------------------------------
    @execution_time
    def update_database_for_interval(
        self, exchange: str, quote_asset: str, interval: str
    ):
        print("-" * 160)
        self.hermes.update_database_for_interval(
            exchange=exchange, quote_asset=quote_asset, interval=interval
        )

    @execution_time
    def check_table(self, symbol: str, interval: str):
        res = self.hermes.check_table(symbol, interval)

        missing = res.get("missing_periods")
        success, failed = res.get("successful"), res.get("failed")
        still_missing = res.get("still_missing")

        for elem in still_missing:
            logger.warning(f"still missing: {missing[elem]}")
        logger.info(f"success: {success}, failed: {failed}")


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == "__main__":

    symbol_name = "BTCUSDT"
    interval = "1d"
    symbols = ["BTCUSDT", "ADAUSDT", "XRPUSDT", "XLMUSDT"]
    intervals = ["5m", "15m", "30m", "1h", "2h", "4h"]

    dates = [
        "1 year ago UTC",
        "now UTC",  # 'October 19, 2023 00:00:00'
    ]
    exchange = "kucoin" if "-" in symbol_name else "binance"

    # -------------------------------------------------------------------------
    ht = HermesTester(exchange=exchange, interval=interval, verbose=True)

    # ht.create_symbols_table()
    # time.sleep(1)
    # ht.drop_symbols_table()
    # ht.update_symbols_table(exchange_name=exchange)
    # ht.write_to_symbols_table()

    # ht.get_tradeable_symbols(quote_asset='BTC', margin_only=False)
    # ht.get_all_symbols_from_exchange(exchange)
    # ht.get_all_symbols_from_database(exchange)

    # ht.update_ohlcv_table(symbol=symbol_name)
    # ht.get_ohlcv_table_status(symbol_name)
    # ht.get_ohlcv_tablename(symbol_name, interval)
    # ht.get_all_tablenames(_filter='1d')
    # ht.get_symbol_and_interval_from()

    # ht.get_symbol(symbol_name)
    # ht.get_multiple_symbols(exchange)
    # ht.get_tradeable_symbols(quote_asset=None)
    # ht.get_leveraged_symbols()

    # ht.get_max_workers()

    # -------------------------------------------------------------------------
    # ht.get_ohlcv_multiple(symbols=symbols, start=dates[0], end=dates[1])

    # for _ in range(3):
    #     ht.get_ohlcv_single(symbol=symbol_name, interval=interval,
    #                         start=dates[0], end=dates[1])

    # -------------------------------------------------------------------------
    ht.update_database_for_interval(
        exchange=exchange, quote_asset='USDT', interval=interval
        )

    # ht.get_listing_date(symbol_name, '15m')

    # ht.check_table(symbol_name, interval)
