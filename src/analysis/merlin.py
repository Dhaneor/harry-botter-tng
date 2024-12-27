#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: Needs to adapted because we do not have the old indicators library anymore.

Created on Sun Jan 31 00:57:06 2021

@author: dhaneor
"""
import pandas as pd
import numpy as np

from exchange.binance_ import Binance
from staff.hermes import Hermes
from data.rawi.util.binance_async_ohlcv import BinanceClient
from src.analysis.indicators.indicators_fast_nb import atr


# =============================================================================
class Merlin:
    """
    Merlin is the wizard who finds the symbols with the highest volatility
    which helps us to find the markets where we can expect profit because
    of strong movements in the price.

    The calculation is based on the 'Average True Range' and can be done for
    different lookback periods.
    """

    def __init__(self, interval: str = "1h"):

        self.symbols: list = []  # list of all symbols that we are interested in
        self.tickers: pd.DataFrame = pd.DataFrame()
        self.klines = []

        self.quote_asset: str = "all"  # 'all' or one specific quote asset
        self.interval = interval
        self.conn: Binance = Binance()
        self.hermes = Hermes(verbose=True)

    def __repr__(self):
        return self.tickers

    # -------------------------------------------------------------------------
    def get_candidates(self, quote_asset: str = "USDT") -> pd.DataFrame:

        self.quote_asset = quote_asset

        self.get_symbols()
        self.get_tickers()
        self._filter_stable_and_fiat()
        # self._filter_based_on_quote_volume()
        # self._filter_based_on_spread(percentile=0.1)

        self.get_klines()
        self._add_indicators_to_klines()
        self._add_atr_percent()
        self._sort_by_atr_volatility()

        return self.tickers

    # -------------------------------------------------------------------------
    # get a list of symbols for a given quote asset (or all of them which
    # probably doesn't make sense most of the time)
    def get_symbols(self):
        self.symbols = self.conn.get_list_of_symbols(quote_asset=self.quote_asset)

    # get all tickers and filter out the ones we need based on our list
    # of symbols
    def get_tickers(self):

        tickers = self.conn.get_all_tickers()["result"]
        df = pd.DataFrame().from_dict(tickers)
        df.drop(["firstId", "lastId"], axis=1, inplace=True)

        df = df[df["count"] > 0]

        for i in range(1, 16):
            df.iloc[:, i] = df.iloc[:, i].astype(float)

        df = self._add_spread_in_percent(df)

        for s in self.symbols:
            self.tickers = self.tickers.append(df[df.symbol == s], ignore_index=True)

    # get the klines (ohclv data) for all the symbols that we are interested in
    def get_klines(self):
        symbols = list(self.tickers["symbol"])
        conn = BinanceClient()
        self.klines = conn.download_ohlcv_data(symbols, self.interval)

    # -------------------------------------------------------------------------
    # sort methods
    #
    # sort by volume (descending)
    def _sort_by_quote_volume(self):
        self.tickers.sort_values(by=["quoteVolume"], ascending=False, inplace=True)

    def _sort_by_price_change(self):
        self.tickers.sort_values(
            by=["priceChangePercent"], ascending=False, inplace=True
        )

    def _sort_by_atr_volatility(self):

        self.tickers = self.tickers.sort_values(by="atr.perc", ascending=False)

    # -------------------------------------------------------------------------
    # filter methods that get rid of values that don't pass the filter
    def _filter_stable_and_fiat(self):

        stable = ["USDCUSDT", "BUSDUSDT"]
        fiat = ["USDUSDT", "EURUSDT", "AUDUSDT", "RUBUSDT"]
        exclude = stable + fiat

        for s in exclude:
            self.tickers = self.tickers[self.tickers["symbol"] != s]

    def _filter_based_on_spread(self, percentile=0.2):
        cutoff = self.tickers["spread percent"].quantile(q=percentile)
        self.tickers = self.tickers[self.tickers["spread percent"] < cutoff]

    def _filter_based_on_quote_volume(self, percentile=0.1):
        cutoff = self.tickers["quoteVolume"].quantile(q=percentile)
        self.tickers = self.tickers[self.tickers["quoteVolume"] > cutoff]

    def _filter_based_on_atr(self):
        pass

    # -------------------------------------------------------------------------
    # methods to add columns with additional (computed) values
    def _add_spread_in_percent(self, df):
        df["spread percent"] = (df["askPrice"] / df["bidPrice"] - 1) * 100
        return df

    def _add_atr_percent(self):

        self.tickers["atr.perc"] = np.nan

        for item in self.klines:
            df = item[0]
            symbol = item[2]

            self.tickers.loc[self.tickers["symbol"] == symbol, "atr.perc"] = df[
                "ATR percent"
            ].iloc[-1]

    def _add_indicators_to_klines(self):
        for idx, kline in enumerate(self.klines):
            self.klines[idx] = (
                self.indicators.average_true_range(
                    df=kline[0], period=720, method="ewm"
                ),
                self.klines[idx][1],
                self.klines[idx][2],
            )
