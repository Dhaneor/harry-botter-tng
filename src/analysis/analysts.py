#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""
import sys
import pandas as pd
import numpy as np
import numpy.typing as npt
import talib as ta
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict

from .indicators.indicators_old import Indicators
from src.plotting.minerva import AnalystChart

i = Indicators()

logger = logging.getLogger("main.analysts")

TimeSeries = Union[List[float], Tuple[float], np.ndarray]


# =============================================================================
class IAnalyst(ABC):

    """
    This is the base class for all strategies which should never be
    instantiated by itself but only inherited by the actual implementations
    of the concrete signal analysts
    """

    def __init__(self):
        self.name: str = "Base Analyst"
        self.comment: str
        self.plot_params: dict = dict()
        self.indicators = Indicators()

    # -------------------------------------------------------------------------
    @abstractmethod
    def get_signal(self, *args, **kwargs):
        pass

    def draw_chart(
        self,
        df: pd.DataFrame,
        subplots: Optional[dict] = None,
        color_scheme: str = "day",
        with_market_state: bool = False,
    ):
        if not subplots:
            subplots = {self.name: self.plot_params}

        c = AnalystChart(
            df=df,
            subplots=subplots,
            color_scheme=color_scheme,
            with_market_state=with_market_state,
        )
        c.draw()

    # -------------------------------------------------------------------------
    def _calculate_slope(self, current_value, previous_value, lookback: int = 1):
        return ((current_value - previous_value) / previous_value) / lookback * 100


# =============================================================================
class RsiAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.name: str = "RSI"
        self.overbought: int = 70
        self.oversold: int = 30

        self.column_name: str = "s.rsi"
        self.comment: str = (
            f"oversold = {self.oversold}, overbought = {self.overbought}"
        )
        self.confidence: float = 0

        self.plot_params = {
            "label": self.name,
            "columns": ["rsi.close.14"],
            "horizontal lines": [self.overbought, self.oversold],
            "channel": [],
            "fill": True,
            "signal": self.column_name,
        }

    def get_signal(self, data: dict, mode: int = 0, lookback: int = 14):
        col = f"rsi.{lookback}"
        self.plot_params["columns"] = [col]

        rsi = data[col] = self.indicators.rsi_ta_lib(data=data["c"], period=lookback)

        prev_rsi = np.roll(rsi, 1)

        if mode == 0:
            conditions = [
                (rsi > prev_rsi) & (prev_rsi < self.oversold),
                (rsi < prev_rsi) & (prev_rsi > self.overbought),
            ]
        elif mode == 1:
            conditions = [rsi < self.oversold, rsi > self.overbought]
        else:
            raise ValueError(f"mode {mode} is not supported")

        choices = [1, -1]
        data["s.rsi"] = np.select(conditions, choices, default=np.nan)

        return data

    # -------------------------------------------------------------------------
    # calculate the confidence level for the RSI signal
    def _get_confidence(self, rsi):
        precision = 2

        if rsi < self.oversold and rsi != 0:
            return round(self.oversold / rsi, precision)

        elif rsi > self.overbought and rsi != 100:
            return round((100 - self.oversold) / (100 - rsi), precision)

        else:
            return 1


class ConnorsRsiAnalyst(IAnalyst):
    def __init__(self):
        self.overbought: int = 95
        self.oversold: int = 5

        self.column_name: str = "s.c_rsi"
        self.comment: str = (
            f"oversold = {self.oversold}, " f"overbought = {self.overbought}"
        )
        self.confidence: float = 0

        self.plot_params = {
            "label": "Connors RSI",
            "columns": [],
            "horizontal lines": [self.overbought, self.oversold],
            "channel": [self.overbought, self.oversold],
            "fill": True,
            "signal": self.column_name,
        }

        super().__init__()

    # -------------------------------------------------------------------------
    # get the signal
    # expects a dataframe with OHLC data and calculates the RSI signal
    # for the last row, if no index is given
    def get_signal(
        self,
        df: pd.DataFrame,
        on_what: str = "close",
        rsi_lookback: int = 2,
        streak_lookback: int = 3,
        smoothing: int = 1,
    ):
        col = f"c_rsi.{on_what.lower()}"
        self.plot_params["columns"] = [col]

        df[col] = self.indicators.connors_rsi(
            data=df[on_what], rsi_lookback=rsi_lookback, streak_lookback=streak_lookback
        )

        df[col] = df[col].rolling(smoothing).mean()

        curr, prev = df[col], df[col].shift()
        # conditions = [(curr > prev) & (prev < self.oversold), \
        #                 (curr < prev) & (prev > self.overbought)]

        conditions = [curr < self.oversold, curr > self.overbought]

        choices = [1, -1]

        df["s.c_rsi"] = np.select(conditions, choices, default=np.nan)
        return df

    # -------------------------------------------------------------------------
    # calculate the confidence level for the RSI signal
    def _get_confidence(self, rsi):
        precision = 2

        if rsi < self.oversold and rsi != 0:
            return round(self.oversold / rsi, precision)

        elif rsi > self.overbought and rsi != 100:
            return round((100 - self.oversold) / (100 - rsi), precision)

        else:
            return 1


class DynamicRateOfChangeAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.overbought: int = 10
        self.oversold: int = -10

        self.column_name: str = "s.droc"
        self.comment: str = (
            f"oversold = {self.oversold}, overbought = {self.overbought}"
        )
        self.confidence: float = 0

        self.plot_params = {
            "label": "TD Rate-Of-Change",
            "columns": ["droc.close"],
            "horizontal lines": [],
            "channel": [],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self,
        df: pd.DataFrame,
        on_what: str = "close",
        lookback: int = 14,
        smoothing: int = 1,
    ) -> pd.DataFrame:
        df = self.indicators.dynamic_rate_of_change(
            df=df,
            on_what=on_what,
            lookback=lookback,
            smoothing=smoothing,
            normalized=False,
        )

        col = f"droc.{on_what.lower()}"
        col_upper, col_lower = f"{col}.upper", f"{col}.lower"

        sig, prev_sig = df[col], df[col].shift()
        ref = df[col].ewm(span=lookback).mean()

        # add standard deviation channel
        window = int(lookback * 1)
        df[col_upper] = ref + sig.rolling(window=window).std()
        df[col_lower] = ref - sig.rolling(window=window).std()
        self.plot_params["channel"] = [col_upper, col_lower]

        upper, prev_upper = df[col_upper], df[col_upper].shift()
        lower, prev_lower = df[col_lower], df[col_lower].shift()

        conditions = [
            (sig > lower) & (prev_sig < prev_lower) & (sig < 0),
            (sig < upper) & (prev_sig > prev_upper) & (sig > 0),
        ]
        choices = [1, -1]

        df["s.droc"] = np.select(conditions, choices, default=np.nan)
        df["droc.close.k"] = ref
        return df

    # -------------------------------------------------------------------------
    def _get_confidence(self, rsi):
        precision = 2

        if rsi < self.oversold and rsi != 0:
            return round(self.oversold / rsi, precision)

        elif rsi > self.overbought and rsi != 100:
            return round((100 - self.oversold) / (100 - rsi), precision)

        else:
            return 1


class FibonacciTrendAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.column_name: str = "s.fib_trend"
        self.comment: str = ""
        self.confidence: float = 0
        self.plot_params = {
            "label": "Fibonacci Trend",
            "columns": ["f_trend", "f_trend.sig"],
            "horizontal lines": [0],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.indicators.fibonacci_trend(df=df)

        trend, prev_trend = df["f_trend"], df["f_trend"].shift()
        sig_line, prev_sig_line = df["f_trend.sig"], df["f_trend.sig"].shift()

        conditions = [
            (trend > sig_line) & (prev_trend <= prev_sig_line),
            (trend < sig_line) & (prev_trend >= prev_sig_line),
        ]

        choices = [1, -1]

        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df

    # -------------------------------------------------------------------------
    # calculate the confidence level for the RSI signal
    def _get_confidence(self, rsi):
        pass


# =============================================================================
class StochRsiAnalyst(IAnalyst):
    def __init__(self):
        self.overbought = 90
        self.oversold = 10

        self.column_name = "s.stc.rsi"
        self.comment = f"oversold = {self.oversold}, overbought = {self.overbought}"

        self.plot_params = {
            "label": "Stochastic RSI",
            "columns": ["stoch.rsi.k", "stoch.rsi.d"],
            "horizontal lines": [self.overbought, self.oversold],
            "channel": [self.overbought, self.oversold],
            "fill": True,
            "signal": self.column_name,
        }

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(
        self,
        df: pd.DataFrame,
        method="crossover",
        period_rsi: int = 14,
        period=14,
        k_period=5,
        d_period=3,
    ):
        df = self.indicators.stoch_rsi(
            df=df,
            period_rsi=period_rsi,
            period=period,
            k_period=k_period,
            d_period=d_period,
        )

        # ---------------------------------------------------------------------
        if method == "extremes":
            col = "stoch.rsi.d"

            conditions = [df[col] < self.oversold, df[col] > self.overbought]
            choices = [1, -1]

        # ---------------------------------------------------------------------
        elif method == "crossover":
            k_line, d_line = df["stoch.rsi.k"], df["stoch.rsi.d"]
            prev_k_line, prev_d_line = (
                df["stoch.rsi.k"].shift(),
                df["stoch.rsi.d"].shift(),
            )
            o_bought, o_sold = self.overbought, self.oversold

            conditions = [
                (k_line > d_line)
                & (prev_k_line <= prev_d_line)
                & (prev_k_line <= o_sold),
                (k_line < d_line)
                & (prev_k_line >= prev_d_line)
                & (prev_k_line >= o_bought),
            ]

            choices = [1, -1]

        else:
            raise ValueError(f"method {method} is not supported")

        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df

    # -------------------------------------------------------------------------
    # calculate the confidence level for the RSI signal
    def _get_confidence(self, val):
        precision = 2

        if val < self.oversold and val != 0:
            return round(self.oversold / val, precision)

        elif val > self.overbought and val != 100:
            return round((100 - self.oversold) / (100 - val), precision)

        else:
            return 1


# =============================================================================
class StochasticAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.overbought = 70
        self.oversold = 30

        self.period = 14
        self.k_period = 5
        self.d_period = 7

        self.column_name = "s.stc"

    @property
    def plot_params(self):
        return {
            "label": "Stochastic",
            "columns": ["stoch.close.d", "stoch.close.k"],
            "horizontal lines": [self.overbought, self.oversold],
            "channel": [self.overbought, self.oversold],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, oversold: int = 30, overbought: int = 70
    ) -> pd.DataFrame:
        self.oversold, self.overbought = oversold, overbought

        df = self.indicators.stochastic(
            df=df, period=self.period, k_period=self.k_period, d_period=self.d_period
        )

        d, k = df["stoch.close.d"], df["stoch.close.k"]
        prev_d, prev_k = df["stoch.close.d"].shift(), df["stoch.close.k"].shift()
        conditions = [
            (k > d) & (prev_k < prev_d) & (prev_k < self.oversold),
            (k < d) & (prev_k > prev_d) & (prev_k > self.overbought),
        ]
        choices = [1, -1]

        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df


# =============================================================================
class CommodityChannelAnalyst(IAnalyst):
    def __init__(self):
        self.overbought = 100
        self.oversold = -100

        self.period = 20

        self.column_name = "s.cci"
        self.comment = f"oversold = {self.oversold}, overbought = {self.overbought}"

        super().__init__()

    @property
    def plot_params(self):
        return {
            "label": "Commodity Channel Index",
            "columns": ["cci"],
            "horizontal lines": [self.overbought, self.oversold],
            "channel": [self.overbought, self.oversold],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    # get the signal
    # expects a dataframe with OHLC data and calculates the RSI signal
    # for the last row, if no index is given
    def get_signal(self, df: pd.DataFrame, period: int = 20, mode=1):
        df = self.indicators.cci(df=df, period=period)

        cci, prev_cci = df["cci"], df["cci"].shift()

        if mode == 1:
            conditions = [
                (cci >= prev_cci) & (prev_cci < self.oversold),
                (cci <= prev_cci) & (prev_cci > self.overbought),
            ]

        elif mode == 2:
            conditions = [cci < self.oversold, cci > self.overbought]

        elif mode == 3:
            conditions = [(cci > 0) & (prev_cci <= 0), (cci < 0) & (prev_cci >= 0)]

        else:
            logger.error(f"mode {mode} is not supported")
            df["self.column_name"] = np.nan
            return df

        choices = [1, -1]

        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df


# =============================================================================
class TrendAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.trnd"
        self.comment: str = ""

        self.ma_type = "sma"  # 'sma' or 'ewm'
        self.ma_short: int = 14
        self.ma_long: int = 42
        self.lookback: int = 1

        super().__init__()

    # -------------------------------------------------------------------------
    # calculate the actual signal, return the updated dataframe and a dictionary
    def get_signal(self, df: pd.DataFrame):
        if self.ma_type == "ewm":
            df = self.indicators.ewma(df=df, period=self.ma_short)
            df = self.indicators.ewma(df=df, period=self.ma_long)

        elif self.ma_type == "sma":
            df = self.indicators.sma(df=df, period=self.ma_short)
            df = self.indicators.sma(df=df, period=self.ma_long)

        # add the 63-day and the 200-day EMAs as trend filter
        df = self.indicators.ewma(df=df, period=63)
        df = self.indicators.ewma(df=df, period=63)

        # ---------------------------------------------------------------------
        col_price = "close"
        col_slp_prc = "slope." + col_price.lower()

        col_short_ma = self.ma_type + "." + str(self.ma_short)
        col_slp_short = "slope." + col_short_ma

        col_long_ma = self.ma_type + "." + str(self.ma_long)
        col_slp_long = "slope." + col_long_ma

        filter_ = df["ewm.63"]

        # ---------------------------------------------------------------------
        df[col_slp_prc] = round((df[col_price] / df[col_price].shift() - 1) * 100, 4)
        df[col_slp_short] = round(
            (df[col_short_ma] / df[col_short_ma].shift() - 1) * 100, 4
        )
        df[col_slp_long] = round(
            (df[col_long_ma] / df[col_long_ma].shift() - 1) * 100, 4
        )

        # conditions = [(df[col_price] > df[col_short_ma]) & (df[col_slp_short] > 0) \
        #                 & (df[col_slp_long] > 0) & (df[col_short_ma] > filter_),
        #               (df[col_price] < df[col_short_ma]) & (df[col_slp_short] < 0) \
        #                   & (df[col_slp_long] < 0)] # & (df[col_short_ma] < filter_)]

        conditions = [
            (df[col_price] > df[col_short_ma])
            & (df[col_slp_short] > 0)
            & (df[col_slp_long] > 0)
            & (df[col_short_ma] > filter_)
            & (df[col_short_ma] > df[col_long_ma]),
            (df[col_price] < df[col_short_ma])
            & (df[col_slp_short] < 0)
            & (df[col_slp_long] < 0)
            & (df[col_short_ma] < filter_)
            & (df[col_short_ma] < df[col_long_ma]),
        ]

        choices = [1, -1]

        df["s.trnd"] = np.select(conditions, choices, default=np.nan)
        # df.loc[df['s.trnd'] == df['s.trnd'].shift(), 's.trnd'] = 0

        # calculate the difference between the two moving averages for later use
        df["ma.diff"] = df[col_short_ma] - df[col_long_ma]

        return df


class BigTrendAnalyst(IAnalyst):
    def __init__(self):
        self.name = "Big Trend Analyst"
        self.column_name = "s.big_trend"
        super().__init__()

        self.plot_params = {
            "main": True,
            "label": "Big Trends",
            "columns": ["bt_mid"],
            "horizontal lines": [],
            "channel": ["bt_upper", "bt_lower"],
            "fill": True,
            "signal": self.column_name,
        }

    def get_signal(
        self,
        df: pd.DataFrame,
        period: int = 20,
        factor: float = 0.001,
        fast_exit: bool = False,
    ) -> pd.DataFrame:
        df["bt_mid"], df["bt_upper"], df["bt_lower"] = self.indicators.big_trend(
            close=df.close, period=period, factor=factor
        )

        if fast_exit:
            # fast exit when close goes is back within bands
            conditions = [
                (df.close > df.bt_upper) & (df.close.shift() <= df.bt_upper.shift()),
                (df.close < df.bt_lower) & ~(df.close.shift() < df.bt_lower.shift()),
                (df.close < df.bt_upper) & ~(df.close.shift() < df.bt_upper.shift()),
                (df.close > df.bt_lower) & ~(df.close.shift() > df.bt_lower.shift()),
            ]
        else:
            # slower exit when close crosses middle line
            conditions = [
                (df.close > df.bt_upper) & (df.close.shift() <= df.bt_upper.shift()),
                (df.close < df.bt_lower) & ~(df.close.shift() < df.bt_lower.shift()),
                (df.close < df.bt_mid) & ~(df.close.shift() < df.bt_mid.shift()),
                (df.close > df.bt_mid) & ~(df.close.shift() > df.bt_mid.shift()),
            ]

        choices = [1, -1, 0, 0]

        df[self.column_name] = np.select(conditions, choices, np.nan)

        return df


# =============================================================================
class MovingAverageCrossAnalyst(IAnalyst):
    """This class produces a signal depending on crossover of a shorter moving
    average and a longer moving average

    The main function 'get_signal()' returns a BUY(1), SELL(-1) or DO NOTHING(0)
    signal based on the detection of a possible crossover to the up- or downside.

    The __init__ function expects two values for the (Simple) Moving Averages to
    be used. The values need to have the format 'SMAx' where x is the number of
    periods for the SMA. The names need to match the column names that are given
    by the 'Indicators' class.
    """

    def __init__(self):
        self.column_name = "s.ma.x"
        self.ma_type = "ewm"
        self.ma_short = 16
        self.ma_long = 64

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(self, data: np.ndarray) -> np.ndarray:
        short, long = self.ma_short, self.ma_long
        short_ma = self.ma_type + "_" + str(short)
        long_ma = self.ma_type + "_" + str(long)

        if self.ma_type in {"ewm", "ewma"}:
            short_ma = ta.EMA(data, timeperiod=self.ma_short)
            long_ma = ta.EMA(data, timeperiod=self.ma_long)
        elif self.ma_type == "sma":
            short_ma = ta.SMA(data, timeperiod=self.ma_short)
            long_ma = ta.SMA(data, timeperiod=self.ma_long)
        else:
            raise ValueError(
                "{self.ma_type} is not a valid type for the moving average."
            )

        # ---------------------------------------------------------------------
        curr = np.subtract(short_ma, long_ma)
        prev = np.subtract(np.roll(short_ma, 1), np.roll(long_ma, 1))

        conditions = [(curr >= 0) & (prev < 0), (curr <= 0) & (prev > 0)]
        choices = [1, -1]

        return np.select(conditions, choices, default=np.nan)

    # -------------------------------------------------------------------------
    # calculate the confidence level for the MA crossover signal
    def _get_confidence(self, sma_short, prev_sma_short):
        return round(abs(self._calculate_slope(sma_short, prev_sma_short)), 2)


class MovingAverageAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.column_name: str = "s.ma"
        self.comment: str = "<none>"

        self.ma_col: str
        self.ma_col_diff: str

    @property
    def plot_params(self):
        return {
            "label": "Moving Average",
            "columns": [self.ma_col_diff],
            "horizontal lines": [],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, period: int = 63, type="ewma"
    ) -> pd.DataFrame:
        if type == "sma":
            ma_col = "sma." + str(period)
            if not ma_col in df.columns:
                df = self.indicators.sma(df=df, period=period)
        else:
            ma_col = "ewm." + str(period)
            if not ma_col in df.columns:
                df = self.indicators.ewma(df=df, period=period)

        self.ma_col, self.ma_col_diff = ma_col, f"{ma_col}.diff"
        df[self.ma_col_diff] = (df["close"] / df[ma_col] - 1) * 100

        # ---------------------------------------------------------------------
        close, ma = df["close"], df[ma_col]
        conditions = [close > ma, close < ma]
        choices = [1, -1]

        df["s.ma"] = np.select(conditions, choices, default=np.nan)
        return df


# =============================================================================
class MomentumAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.mom"

        self.overbought: float = 10
        self.oversold: float = -1 * self.overbought

        self.short_sma: int = 3
        self.long_sma: int = 7
        self.lookback = 9

        self.comment: str = (
            f"oversold = {self.oversold}, overbought = {self.overbought}"
        )

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        col_mom = "mom"
        col_short = col_mom + "." + "sma." + str(self.short_sma)
        col_long = col_mom + "." + "sma." + str(self.long_sma)

        if not col_mom in df.columns:
            df = self.indicators.momentum(df=df, lookback=self.lookback)

        if not col_short in df.columns:
            df = self.indicators.sma(
                df=df, on_what=col_mom, period=self.short_sma, col_prefix="mom."
            )

        if not col_long in df.columns:
            df = self.indicators.sma(
                df=df, on_what=col_mom, period=self.long_sma, col_prefix="mom."
            )

        # ---------------------------------------------------------------------
        mom, short_ma, long_ma = df["mom"], df[col_short], df[col_long]
        prev_mom, prev_short_ma, prev_long_ma = (
            mom.shift(),
            short_ma.shift(),
            long_ma.shift(),
        )

        # conditions = [(short_ma > long_ma) & (prev_short_ma <= prev_long_ma) & (prev_short_ma < self.oversold),
        #               (short_ma < long_ma) & (prev_short_ma >= prev_long_ma) & (prev_short_ma > self.overbought)]

        conditions = [
            (short_ma > long_ma) & (prev_short_ma <= prev_long_ma),
            (short_ma < long_ma) & (prev_short_ma >= prev_long_ma),
        ]

        # conditions = [(short_ma > long_ma) & (prev_long_ma < 10),
        #             (long_ma < 10) & (prev_long_ma == 10)]

        choices = [1, -1]

        df["s.mom"] = np.select(conditions, choices, default=np.nan)

        return df


# =============================================================================
class KeltnerChannelAnalyst(IAnalyst):
    def __init__(self, mode=1):
        self.column_name: str = "s.kc"
        self.mode = mode

        self.kc_lookback = 14
        self.atr_lookback = 14

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(
        self,
        df: Optional[pd.DataFrame] = None,
        data_as_dict: Optional[Dict[str, np.ndarray]] = None,
        multiplier: float = 2,
        period: int = 20,
        mode: int = 0,
    ) -> Union[pd.DataFrame, dict, None]:
        """Calculates the signal based on Keltner Channel analysis.

        Price data can be supplied either as a dataframe or as a dictionary.

        :param df: OHLCV dataframe
        :type df: pd.DataFrame
        :param data_as_dict: dictionary of OHLCV data, defaults to None
        :type data_as_dict: dict, optional
        :param multiplier: multiplier Keltner channel, defaults to 2.5
        :type multiplier: float, optional
        :param period: KC/ATR lookback, defaults to 20
        :type period: int, optional
        :param mode: defines analysis mode, defaults to 0

        0: contrarian signal triggered by close outside KC (exit: close
        crossing MA)

        1: trend following signal triggered by close outside KC (close
        at new signal or when SL is triggered - needs trailing SL)

        2: trend following signal triggered by close outside KC (exit:
        down candle after candle with close outside KC)

        :type mode: int, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        if df is not None:
            keltner = self.indicators.keltner_talib

            df["kc.upper"], df["kc.lower"], df["kc.mid"] = keltner(
                open_=df.open.to_numpy(),
                high=df.high.to_numpy(),
                low=df.low.to_numpy(),
                close=df.close.to_numpy(),
                period=period,
                atr_multiplier=multiplier,
                atr_lookback=20,
            )

            close = df.close
            prev_close = df.close.shift(1)
            upper_limit, lower_limit = df["kc.upper"], df["kc.lower"]
            prev_upper_limit = df["kc.upper"].shift(1)
            prev_lower_limit = df["kc.lower"].shift()
            mid, prev_mid = df["kc.mid"], df["kc.mid"].shift(1)

        elif data_as_dict is not None:
            keltner = self.indicators.keltner_talib
            d = data_as_dict

            kc_upper, kc_lower, kc_mid = keltner(
                open_=d["o"],
                high=d["h"],
                low=d["l"],
                close=d["c"],
                period=period,
                atr_multiplier=multiplier,
            )

            close = d["c"]
            prev_close = np.roll(close, 1)
            upper_limit, lower_limit = kc_upper, kc_lower
            prev_upper_limit = np.roll(kc_upper, 1)
            prev_lower_limit = np.roll(kc_lower, 1)
            mid, prev_mid = kc_mid, np.roll(kc_mid, 1)

        # set conditions for signal based on 'mode' parameter
        if mode == 0:
            conditions = [
                (close > upper_limit),
                (close <= mid) & (prev_close > prev_mid),
                (close >= mid) & (prev_close < prev_mid),
                close < lower_limit,
            ]

            choices = [-1, 0, 0, 1]

        elif mode == 1:
            conditions = [
                (close > upper_limit) & (prev_close <= prev_upper_limit),
                (close <= mid) & (prev_close > prev_mid),
                (close >= mid) & (prev_close < prev_mid),
                (close < lower_limit) & (prev_close >= prev_lower_limit),
            ]

            choices = [1, 0, 0, -1]

        elif mode == 2:
            conditions = [
                (close > upper_limit) & (prev_close <= prev_upper_limit),
                (close < prev_close) & (prev_close > prev_upper_limit),
                (close > prev_close) & (prev_close < prev_lower_limit),
                (close < lower_limit) & (prev_close >= prev_lower_limit),
            ]

            choices = [1, 0, 0, -1]

        else:
            raise ValueError(f"mode must be 0, 1 or 2, but was {mode}")

        signal = np.select(conditions, choices, default=np.nan)

        if df is not None:
            df["s.kc"] = signal
            return df
        elif data_as_dict is not None:
            data_as_dict["s.kc"] = signal
            return data_as_dict


# =============================================================================
class AverageDirectionalIndexAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.adx"
        super().__init__()

        self.threshhold: float = 25

        self.plot_params = {
            "label": "ADX",
            "columns": ["adx", "adx.di.pls", "adx.di.mns"],
            "horizontal lines": [self.threshhold],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    # get the signal
    # expects a dataframe with OHLC data and calculates the RSI signal
    # for the last row, if no index is given
    def get_signal(
        self, df: pd.DataFrame, lookback: int = 14, threshhold: int = 25
    ) -> pd.DataFrame:
        self.threshhold = threshhold

        if not "adx" in df.columns:
            i = Indicators()
            df = i.adx(df=df, lookback=lookback)

        adx = df["adx"]
        plus_di, minus_di = df["adx.di.pls"], df["adx.di.mns"]

        # ---------------------------------------------------------------------
        conditions = [
            (adx > threshhold) & (plus_di > minus_di),
            (adx > threshhold) & (plus_di < minus_di),
        ]

        choices = [1, -1]
        df["s.adx"] = np.select(conditions, choices, default=np.nan)

        # ---------------------------------------------------------------------
        # for col in df.columns:
        #     if 'Volume' in col: df.drop(col, axis=1, inplace=True)
        # df.drop(['Close Time', 'Number of Trades'], axis=1, inplace=True)

        # # df = df[df['s.kc'] < 0]
        # print(df.tail(50))
        # sys.exit()

        return df


class ChoppinessIndexAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.column_name: str = "s.ci"

        self.plot_params = {
            "label": "Choppiness Index",
            "columns": ["ci"],
            "horizontal lines": [38.2, 61.8],
            "channel": [38.2, 61.8],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, lookback: int = 14, atr_lookback: int = 14
    ) -> pd.DataFrame:
        df = self.indicators.choppiness_index(
            df=df, lookback=lookback, atr_lookback=atr_lookback
        )

        ci = df["ci"]
        ranging = 61.8
        trending = 50  # 38.2

        df["sma.64"] = ta.SMA(df.close, 64)

        # ---------------------------------------------------------------------
        conditions = [ci > ranging, ci < trending]
        choices = [0, 1]
        df["s.ci"] = np.select(conditions, choices, default=0)

        # ---------------------------------------------------------------------
        for col in df.columns:
            if "volume" in col:
                df.drop(col, axis=1, inplace=True)

        return df


class NoiseAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.noise"
        super().__init__()

        self.ranging = 0.2

        self.plot_params = {
            "label": "Noise",
            "columns": ["noise"],
            "horizontal lines": [self.ranging],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, period: int = 14, smoothing: int = 14
    ) -> pd.DataFrame:
        df["noise"] = self.indicators.noise_index(
            data=df.close, period=period, smoothing=smoothing
        )

        df["sma.64"] = ta.SMA(df.close, 64)

        # ---------------------------------------------------------------------
        conditions = [df.noise >= self.ranging, df.noise < self.ranging]
        choices = [1, 0]

        df["s.noise"] = np.select(conditions, choices, default=0)

        return df


class TrendyAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.trendy"
        super().__init__()

    @property
    def plot_params(self) -> dict:
        return {
            "label": "trendy",
            "columns": ["trendy", "trendy.sm"],
            "horizontal lines": [],
            "channel": ["trendy.upper", "trendy.lower"],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    # get the signal
    # expects a dataframe with OHLC data and calculates the CI
    # signal for the last row, if no index is given
    def get_signal(
        self,
        df: pd.DataFrame,
        lookback: int = 14,
        smoothing: int = 1,
        threshhold: float = 2,
    ) -> pd.DataFrame:
        if not "trendy" in df.columns:
            i = self.indicators
            df["trendy"] = i.trendy_index(data=df["close"], lookback=lookback)

        if smoothing > 1:
            df.trendy = df.trendy.rolling(window=smoothing).mean()

        trendy = df.trendy
        smoothed = trendy.rolling(window=10).mean()

        std = trendy.rolling(window=lookback).std()
        upper = std * threshhold
        lower = std * threshhold * -1

        df["trendy.sm"] = smoothed
        df["trendy.upper"] = upper
        df["trendy.lower"] = lower

        # ---------------------------------------------------------------------
        # conditions = [df.trendy > threshhold, df.trendy < (threshhold * -1)]
        conditions = [
            (trendy > smoothed)
            & (trendy.shift() <= (smoothed.shift() * -1))
            & (smoothed < lower),
            (trendy < smoothed)
            & (trendy.shift() >= (smoothed.shift() * -1))
            & (smoothed > upper),
        ]

        choices = [1, -1]

        # conditions = [
        #     (trendy > upper) & (trendy.shift() <= (upper.shift())),
        #     (trendy < lower) & (trendy.shift() >= (lower.shift())),
        #     (trendy < upper) & (trendy.shift() >= (upper.shift())),
        #     (trendy > lower) & (trendy.shift() <= (lower.shift())),
        # ]

        # choices = [1, -1, 0, 0]
        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df


class AtrMomentumAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.name = "ATR Momentum"
        self.column_name: str = "s.atr.mom"

        self.plot_params = {
            "label": "ATR",
            "columns": ["atr.mom", "atr.smooth"],
            "horizontal lines": [],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        if not "atr" in df.columns:
            i = Indicators()
            df = i.average_true_range(df=df, period=lookback)  # type:ignore

        atr_mom = df["atr"].pct_change().ewm(span=3).mean()
        df["atr.smooth"] = atr_mom.rolling(window=7).mean()

        # ---------------------------------------------------------------------
        conditions = [atr_mom > 0, atr_mom < 0]
        choices = [1, -1]

        df["atr.mom"] = atr_mom
        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        return df


# =============================================================================
# candle and candlestick pattern analysts
class WickAndBodyAnalyst(IAnalyst):
    def __init__(self):
        self.column_name: str = "s.wab"
        self.comment: str = "Compares wick to body"

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, factor: float = 2, confirmation: bool = True
    ) -> pd.DataFrame:
        # if confirmation is set to TRUE, look for the candlestick from
        # the last period and check if this candle confirms the
        # anticipated change in direction
        if confirmation:
            upper_wick = abs(
                df["high"].shift() - df[["open", "close"]].shift().max(axis=1)
            )
            lower_wick = abs(
                df[["open", "close"]].shift().min(axis=1) - df["low"]
            ).shift()
            body = abs(df["close"] - df["open"].shift()) * 2

            # make series that shows if we went up/down for every interval
            conditions = [df["close"] > df["open"], df["close"] < df["open"]]
            choices = [1, -1]
            direction = np.select(conditions, choices, default=np.nan)

            # find the matching candlesticks (w/ confirmation)
            conditions = [
                (upper_wick > body) & (upper_wick > lower_wick) & (direction == -1),
                (lower_wick > body) & (upper_wick < lower_wick) & (direction == 1),
            ]
            choices = [-1, 1]

            df[self.column_name] = np.select(conditions, choices, default=np.nan)

        # ---------------------------------------------------------------------
        else:
            upper_wick = abs(df["high"] - df[["open", "close"]].max(axis=1))
            lower_wick = abs(df[["open", "close"]].min(axis=1) - df["low"])
            body = abs(df["close"] - df["open"]) * factor

            # find the matching candlesticks (w/o confirmation)
            conditions = [
                (upper_wick > body) & (upper_wick > lower_wick),
                (lower_wick > body) & (upper_wick < lower_wick),
            ]
            choices = [-1, 1]

            df[self.column_name] = np.select(conditions, choices, default=np.nan)
        return df


class BreakoutAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.column_name: str = "s.bo"
        self.comment: str = "Searches for breakouts"
        self.variable_lookback: bool = False

        self.plot_params = {
            "label": "Breakout",
            "columns": ["close", "bo.hh", "bo.ll"],
            "horizontal lines": [],
            "channel": [],
            "fill": False,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame, lookback: int = 28) -> pd.DataFrame:
        if self.variable_lookback:
            df = self._find_breakouts_with_variable_lookback(df, lookback)
        else:
            df["bo.looback"] = lookback
            df["bo.hh"] = df["close"].shift().rolling(lookback).max()
            df["bo.ll"] = df["close"].shift().rolling(lookback).min()

        col = self.column_name

        # find the matching candlesticks (w/o confirmation)
        conditions = [(df["close"] > df["bo.hh"]), (df["close"] < df["bo.ll"])]
        choices = [1, -1]

        df[col] = np.select(conditions, choices, default=np.nan)
        df.loc[df[col] == df[col].shift(), col] = np.nan

        return df

    def _find_breakouts_with_variable_lookback(
        self, df: pd.DataFrame, lookback: int
    ) -> pd.DataFrame:
        df["bo.lookback"] = self._get_variable_lookback(df, lookback)
        lookbacks = list(set(df["bo.lookback"].to_list()))

        for lb in lookbacks:
            df[f"max.{lb}"] = df["close"].shift().rolling(lb).max()
            df[f"min.{lb}"] = df["close"].shift().rolling(lb).min()

            df.loc[df["bo.lookback"] == lb, "bo.hh"] = df[f"max.{lb}"]
            df.loc[df["bo.lookback"] == lb, "bo.ll"] = df[f"min.{lb}"]

            df.drop([f"max.{lb}", f"min.{lb}"], axis=1, inplace=True)

        return df

    def _get_variable_lookback(self, df, base_lookback) -> pd.Series:
        short_atr = i.average_true_range(
            open_=df["open"],
            high_=df["high"],
            low_=df["low"],
            close_=df["close"],
            period=7,
        )
        long_atr = i.average_true_range(
            open_=df["open"],
            high_=df["high"],
            low_=df["low"],
            close_=df["close"],
            period=60,
        )
        factor = long_atr / short_atr
        factor.ffill(inplace=True)
        factor.replace(np.nan, base_lookback)

        variable_lookback = (base_lookback * factor).astype(int)

        return variable_lookback


class TDClopwinPatternAnalyst(IAnalyst):
    """Searches for the Tom Demark Clopwin pattern

    Long signal:
    - The open and close of the current price bar must be contained
        within the open and close range of the previous price bar.
    - The close of the current price bar must be above the close of
        the prior price bar.

    Short signal:
    - The open and close of the current price bar must be contained
        within the open and close range of the previous price bar.
    - The close of the current price bar must be below the close of
        the prior price bar.
    """

    def __init__(self):
        self.column_name: str = "s.td_clop"
        self.comment: str = "Searches for breakouts"

        super().__init__()

    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        curr_open, curr_close = df["open"], df["close"]
        prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
        prior_close = df["close"].shift(2)

        conditions = [
            (prev_open < curr_open)
            & (curr_open <= prev_close)
            & (prev_open < curr_close)
            & (curr_close < prev_close)
            & (curr_close > prior_close),
            (prev_open > curr_open)
            & (prev_open > curr_close)
            & (curr_open >= prev_close)
            & (curr_close > prev_close)
            & (curr_close < prior_close),
        ]
        choices = [1, -1]
        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        if not "ewm.20" in df.columns:
            df = self.indicators.ewma(df=df, period=20)

        ewm = df["ewm.20"]

        df.loc[(df[self.column_name] == -1) & (df["close"] < ewm), self.column_name] = 0
        df.loc[(df[self.column_name] == 1) & (df["close"] > ewm), self.column_name] = 0

        # print(df.loc[df[self.column_name] != 0].tail(50))
        # print(df.tail(50))
        # sys.exit()
        # df.loc[df[col] == df[col].shift(), col] = 0

        return df


class DisparityAnalyst(IAnalyst):
    def __init__(self):
        super().__init__()
        self.column_name: str = "s.disp"
        self.comment: str = "Analyzes the disparity between two values"

        self.column_a = "close"
        self.column_b = None

        self.plot_params = {
            "label": "Disparity",
            "columns": ["disp"],
            "horizontal lines": [0, 0],
            "channel": ["disp.upper", "disp.lower"],
            "fill": True,
            "signal": self.column_name,
        }

    # -------------------------------------------------------------------------
    def get_signal(
        self, df: pd.DataFrame, lookback_periods: List[int] = [21, 200, 500]
    ) -> pd.DataFrame:
        col_a = f"ema.{lookback_periods[0]}"
        col_b = f"ema.{lookback_periods[1]}"
        col_c = f"ema.{lookback_periods[2]}"

        df[col_a] = ta.EMA(df.close, lookback_periods[0])
        df[col_b] = ta.EMA(df.close, lookback_periods[1])
        df[col_c] = ta.EMA(df.close, lookback_periods[2])

        # ......................................................................
        disp = self.indicators.disparity_index(
            df=df, column_a=df[col_a], column_b=df[col_b], column_c=df[col_c]
        )

        disp_mean = disp.ewm(span=14).mean()
        disp_std = disp_mean.rolling(14).std()

        upper = disp_mean + 1 * disp_std
        lower = disp_mean - 1 * disp_std

        rolling_max = disp.rolling(14).max()
        rolling_min = disp.rolling(14).min()

        conditions = [(disp == rolling_max), (disp == rolling_min)]

        # conditions = [
        #     (disp < upper) & (disp.shift() > upper.shift()),
        #     (disp > disp_mean) & (disp.shift() < disp_mean.shift())
        # ]

        choices = [1, -1]

        df["disp"] = disp
        df["disp.upper"] = upper
        df["disp.lower"] = lower

        df[self.column_name] = np.select(conditions, choices, default=np.nan)

        # df.loc[
        #     df[self.column_name] == df[self.column_name].shift(),
        #     self.column_name
        # ] = np.nan

        return df


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #

if __name__ == "__main__":
    pass
