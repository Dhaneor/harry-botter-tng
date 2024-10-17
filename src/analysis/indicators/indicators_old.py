#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 23:12:50 2021

@author: dhaneor
"""
import pandas as pd
import numpy as np
import logging
import talib as ta

from numba import jit
from typing import Union, List, Tuple, Optional, Iterable
import numpy.typing as npt

from src.helpers.fibonacci import fibonacci_series
from src.helpers.timeops import execution_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ==============================================================================
TimeSeries = Union[List[float], Tuple[float], np.ndarray]


class Indicators:
    # ------------------------------------------------------------------------------
    #                                   BASIC STATS                                #
    # ------------------------------------------------------------------------------
    # return per interval
    def returns(self, on_what: pd.Series) -> pd.Series:
        return on_what.pct_change()

    def log_returns(self, on_what: pd.Series) -> pd.Series:
        """Calculates the log returns for a time series of prices.

        :param on_what: a Pandas Series with prices
        :type on_what: pd.Series
        :return: a Pandas Series with the log returns for the prices
        :rtype: pd.Series
        """
        return on_what.apply(np.log).diff(1)

    def average_returns(self, on_what: pd.Series, lookback: int = 21) -> pd.Series:
        """Calculates the average returns for a times series over
        a lookback period.

        :param on_what: a pandas series with prices
        :type on_what: pd.Series
        :param lookback: the lookback period for the average,
        defaults to 21
        :type lookback: int, optional
        :return: a Pandas Series with the average returns
        :rtype: pd.Series
        """
        returns = self.returns(on_what=on_what)
        return returns.ewm(span=lookback).mean()

    def average_true_range(
        self,
        df: Optional[pd.DataFrame] = None,
        open_: Optional[pd.Series] = None,
        high_: Optional[pd.Series] = None,
        low_: Optional[pd.Series] = None,
        close_: Optional[pd.Series] = None,
        period: int = 14,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Calculates the AVERAGE TRUE RANGE for an OHLCV array.

        The caller can provide a DataFrame or four Pandas series and the
        returned type changes accordingly. If both are provided, the
        DataFrame takes precendence.

        :param df: A Pandas DataFrame with 'open', 'high', 'low', 'close'
        :type df: pd.DataFrame, optional
        :param open_: open prices, defaults to None
        :type open_: pd.Series, optional
        :param high_: high prices, defaults to None
        :type high_: pd.Series, optional
        :param low_: low prices, defaults to None
        :type low_: pd.Series, optional
        :param close_: close prices, defaults to None
        :type close_: pd.Series, optional
        :param period: lookback period, defaults to 14
        :type period: int, optional
        :raises ValueError: error if neither DataFrame or complete
        open/high/low/prices are provided
        :return: the DataFrame (column: 'atr') or Series with the
        ATR values
        :rtype: pd.DataFrame or pd.Series
        """
        if df is None:
            return_as_dataframe = False
            if any(arg is None for arg in (open_, high_, low_, close_)):
                raise ValueError(
                    f"Please provide value series for all of: "
                    f"open, high, low, close (or a dataframe "
                    f"with columns: 'open', 'high', 'low', 'close)!"
                )
        else:
            return_as_dataframe = True
            open_, high_ = df["open"], df["high"]
            low_, close_ = df["low"], df["close"]

        # .....................................................................
        high_low = np.abs(high_ - low_)  # type:ignore
        low_close = np.abs(low_ - close_.shift(1))  # type:ignore
        high_close = np.abs(high_ - close_.shift(1))  # type:ignore
        ranges = pd.concat([high_low, high_close, low_close], axis=1)

        true_range = np.max(ranges, axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        if df is not None and return_as_dataframe:
            df["atr"] = atr
            return df
        else:
            return atr

    # -----------------------------------------------------------------------------
    #                                   MOVING AVERAGES                           #
    # -----------------------------------------------------------------------------

    def sma(
        self,
        df: pd.DataFrame,
        period: int,
        on_what: str = "close",
        col_prefix: str = "",
    ) -> pd.DataFrame:
        """Adds a column with the Simple Moving Average (SMA) to a dataframe.

        :param df: a Dataframe containing at least a 'close' column
        (or the column specified in 'on_what' parameter)
        :type df: pd.DataFrame
        :param period: lookback in periods/intervals
        :type period: int
        :param on_what: the column to get the MA for, defaults to 'close'
        :type on_what: str, optional
        :param col_prefix: prefix for result column/series, defaults to ''
        :type col_prefix: str, optional
        :raises ValueError: raise Error if on_what column is missing
        :return: the dataframe with the added result column
        :rtype: Dataframe
        """
        column = col_prefix + "sma." + str(period)
        self._qp = self._get_precision(df["close"])
        if on_what in df.columns:
            df[column] = (
                df[on_what].rolling(period).mean().round(self._qp).astype(float)
            )
            return df
        else:
            raise ValueError(f"{on_what} is not a valid column!")

    def ewma(
        self,
        df: pd.DataFrame,
        period: int,
        on_what: str = "close",
    ) -> pd.DataFrame:
        """Adds a column with the Exponentially Weighted Movin Average
        (EWMA) to a dataframe.

        :param df: a Dataframe containing at least a 'close' column
        (or the column specified in 'on_what' parameter)
        :type df: pd.DataFrame
        :param period: lookback in periods/intervals
        :type period: int
        :param on_what: the column to get the MA for, defaults to 'close'
        :type on_what: str, optional
        :param col_prefix: prefix for result column/series, defaults to ''
        :type col_prefix: str, optional
        :raises ValueError: raise Error if on_what column is missing
        :return: the dataframe with the added result column
        :rtype: Dataframe
        """
        column_name = "ewm." + str(period)

        if on_what in df.columns:
            df[column_name] = (
                df[on_what]
                .ewm(span=period)
                .mean()
                .round(self._get_precision(df["close"]))
            )
            return df
        else:
            raise ValueError(f"{on_what} is not a valid column!")

    # -----------------------------------------------------------------------------
    #                                 MOMENTUM INDICATORS                         #
    # -----------------------------------------------------------------------------

    def rsi(self, data: pd.Series, lookback: int = 14) -> pd.Series:
        """Adds a column with the Relative Strength Index (RSI) to a dataframe.

        :param df: a Dataframe containing at least a 'close' column
        (or the column specified in 'on_what' parameter)
        :type df: pd.DataFrame
        :param on_what: the column to get the RSI for, defaults to 'close'
        :type on_what: str, optional
        :param lookback: the lookback period for the RSI, defaults to 14
        :type lookback: int
        :param column: the column name for the result column
        :type column: str, optional
        :return: _description_
        :rtype: _type_
        """
        delta = data.diff()
        ema_up = delta.clip(lower=0).ewm(span=lookback, adjust=False).mean()
        ema_down = (-1 * delta.clip(upper=0)).ewm(span=lookback, adjust=False).mean()
        rs = ema_up / ema_down

        return 100 - (100 / (1 + rs))

    def rsi_ta_lib(self, data: TimeSeries, period: int = 14) -> npt.NDArray[np.float64]:
        return ta.RSI(data, timeperiod=period)

    def connors_rsi(
        self, data: pd.Series, rsi_lookback: int = 2, streak_lookback: int = 3
    ) -> pd.Series:
        rsi = self.rsi(data, lookback=rsi_lookback)

        streak = self.streak_duration(data)
        rsi_streak = self.rsi(data=streak, lookback=streak_lookback)

        delta = data.diff()
        rank = delta.rolling(100).apply(lambda x: pd.Series(x).rank(pct=False).iloc[-1])

        connors_rsi = (rsi + rsi_streak + rank) / 3

        return connors_rsi

    def stochastic(
        self,
        df: pd.DataFrame,
        on_what: str = "close",
        period: int = 5,
        k_period: int = 3,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Calculate the Stochastic indocator.

        This usually makes sense for the 'close', but
        also for RSI for instance. The method expects

        df  ::  the original dataframe
        on_what   :: name of the column to compute the Stochastic for
        k-period    ::  the lookback for the K-line
        d-period    ::  the lookback for the D-Line
        period  ::  the lookback for the Stochastic
        """

        if on_what not in df.columns:
            raise ValueError(f"{on_what} is not a valid column!")

        # define the result column names
        res_col = on_what.split(".")[0].lower()
        stoch_col = f"stoch.{res_col}"
        k_col = f"{stoch_col}.k"
        d_col = f"{stoch_col}.d"
        diff_col = f"{stoch_col}.diff"

        # calculate the values
        min_val = df[on_what].rolling(window=period, center=False).min()
        max_val = df[on_what].rolling(window=period, center=False).max()

        df[stoch_col] = ((df[on_what] - min_val) / (max_val - min_val)) * 100
        df[k_col] = df[stoch_col].rolling(window=k_period, center=False).mean()
        df[d_col] = df[k_col].rolling(window=d_period, center=False).mean()
        df[diff_col] = df[k_col] - df[d_col]

        return df

    def stoch_rsi(
        self,
        df: pd.DataFrame,
        period_rsi: int = 14,
        period: int = 5,
        k_period: int = 3,
        d_period: int = 3,
    ) -> pd.DataFrame:
        rsi_column = "rsi.close.{str(period_rsi)}"

        if not rsi_column in df.columns:
            df[rsi_column] = self.rsi(data=df["close"], lookback=period_rsi)

        return self.stochastic(
            df=df,
            on_what=rsi_column,
            period=period,
            k_period=k_period,
            d_period=d_period,
        )

    def momentum(
        self, df: pd.DataFrame, lookback: int = 14, on_what: str = "close"
    ) -> pd.DataFrame:
        """Adds a column with the Momentum indicatror to a dataframe.

        :param df: a Dataframe containing at least a 'close' column
        (or the column specified in 'on_what' parameter)
        :type df: pd.DataFrame
        :param lookback: the lookback period, defaults to 14
        :type lookback: int, optional
        :param on_what: the column to get the MA for, defaults to 'close'
        :type on_what: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        if not on_what in df.columns:
            raise ValueError(f"{on_what} is not a valid column!")

        df["mom"] = (
            (((df[on_what] / df[on_what].shift(lookback)) * 100) - 100)
            .clip(lower=-10, upper=10, inplace=False)
            .round(4)
        )

        return df

    def streak_duration(self, data: pd.Series) -> pd.Series:
        # TODO  find a solution that doesn't use a loop!

        df = pd.DataFrame(columns=["data"], data={"data": data})

        df["change"] = df["data"].diff()

        conditions = [data > data.shift(), data < data.shift()]
        choices = [1, -1]
        df["streaks"] = np.select(conditions, choices, default=0)

        df["up"] = df["data"] > df["data"].shift()
        df["down"] = df["data"] < df["data"].shift()
        df["crossing"] = df["streaks"] != df["streaks"].shift()

        res, counter = [], 1
        up = df["up"].tolist()
        down = df["down"].tolist()
        crossing = df["crossing"].tolist()

        for idx in range(len(df["up"])):
            if crossing[idx]:
                counter = 1 if up[idx] else -1
            else:
                counter += 1 if up[idx] else -1

            res.append(counter)

        return pd.Series(res)

    def dynamic_rate_of_change(
        self,
        df: pd.DataFrame,
        on_what="close",
        lookback: int = 14,
        smoothing: int = 1,
        normalized: bool = False,
    ) -> pd.DataFrame:
        col_name = f"droc.{on_what}"
        df[col_name] = self._rate_of_change(df, lookback, on_what)

        if normalized:
            df[col_name] = self._normalize_values(df[col_name], lookback)

        if smoothing > 1:
            df[col_name] = df[col_name].rolling(smoothing).mean().round(4).astype(float)

        return df

    def _rate_of_change(
        self, df: pd.DataFrame, lookback: int = 1, on_what="close"
    ) -> pd.Series:
        if not on_what in df.columns:
            raise ValueError(f"{on_what} is not a valid column!")

        shifted = df[on_what].shift(lookback)
        return ((df[on_what] - shifted) / shifted) * 100

    def fibonacci_trend(
        self, df: pd.DataFrame, max_no_of_periods: int = 100, smoothing: int = 9
    ) -> pd.DataFrame:
        """Calculates the Fibonacci Trend indicator.

        :param df: OHLCV dataframe
        :type df: pd.DataFrame
        :param max_no_of_periods: use Fibonacci numbers smaller than
        this parameter as lookback for exponential moving averages,
        defaults to 100
        :type max_no_of_periods: int, optional
        :param smoothing: smoothing factor for Fibonacci
        trend line, defaults to 9
        :type smoothing: int, optional
        :return: DataFrame with two additional columns for Fibonacci
        trend line and signal line
        :rtype: pd.DataFrame
        """
        # create a temporary dataframe that contains the values for the
        # exponential moving averages for every lookback from the
        # Fibonacci series, where lookback is smaller than the value
        # given in 'max_no_of_periods' parameter
        _df = pd.DataFrame()
        close = df.close.to_numpy()

        for p in fibonacci_series(max_no_of_periods):
            if p > 1:
                _df[f"ewm.{p}"] = ta.SMA(close, timeperiod=p)

        # calculate the trend line
        df["f_trend"] = (
            _df.pct_change(axis=1).ewm(span=smoothing).mean().sum(axis=1) * -1
        )

        # calculate the signal line
        df["f_trend.sig"] = ta.EMA(df.f_trend, 14)

        return df

    @execution_time
    def fibonacci_trend_nb(
        self, df: pd.DataFrame, max_no_of_periods: int = 100, smoothing: int = 9
    ) -> pd.DataFrame:
        """Calculates the Fibonacci Trend indicator using numpy.

        NOTE:   This is my attempt to use numpy for this function. However,
                I was not able to achieve significant speedup and because of
                a slightly different way of calculating the Exponential Moving
                Averages (used heavily here) it gives significantly different
                results than the Pandas implementation above ... should not
                be used as it is!

        :param df: _description_
        :type df: pd.DataFrame
        :param max_no_of_periods: _description_, defaults to 100
        :type max_no_of_periods: int, optional
        :param smoothing: _description_, defaults to 9
        :type smoothing: int, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        # helper functions
        def ema(data, window):
            weights = np.exp(np.linspace(-1.0, 0.0, window))
            weights /= weights.sum()
            a = np.convolve(data, weights, "full")[: len(data)]
            a[:window] = a[window]
            return a

        def ema_numpy(data, window, adjust=True):
            weights = np.ones(window) / window  # Initial weights
            out = np.full_like(data, np.nan)

            out[0] = data[0]  # EMA(0) = value(0)

            for i in range(1, len(data)):
                c = data[i]  # Current value
                w = weights[0]  # First weight

                # Adjust weighting window
                weights = np.concatenate([weights[1:], [1 / window]])

                if adjust:
                    w = 1 - (1 - w) ** (window / (window + 1))  # Adjust fist weight

                ema_prev = out[i - 1]  # Prior EMA

                out[i] = c * w + ema_prev * (1 - w)

            return out

        def ema_2d(data, window, axis=0):
            weights = np.exp(np.linspace(-1.0, 0.0, window))
            weights /= weights.sum()

            if axis == 0:
                a = np.array(
                    [
                        np.convolve(data[:, i], weights, mode="full")[window:-window]
                        for i in range(data.shape[1])
                    ]
                ).T
            else:
                a = np.array(
                    [
                        np.convolve(data[i, :], weights, mode="full")[window:-window]
                        for i in range(data.shape[0])
                    ]
                )

            a[:window] = a[window]
            return a

        def np_pct_change(arr, axis=0):
            diff = np.diff(arr, axis=axis)
            prev = arr[:-1] if axis == 0 else arr[:, :-1]
            return diff / prev

        # ----------------------------------------------------------------------
        close = df.close.to_numpy()
        fib_series = fibonacci_series(max_no_of_periods)
        temp = np.zeros(shape=(close.shape[0], len(fib_series)))

        for col, p in enumerate(fib_series):
            if p > 1:
                temp[:, col] = ta.SMA(close, timeperiod=p)
                # temp[:, col] = ta.EMA(close, p)

        temp = np_pct_change(temp, axis=1)
        temp = ema_2d(temp, window=smoothing, axis=0)

        trend = np.sum(temp, axis=1) * -1

        len_orig, len_now = close.shape[0], temp.shape[0]
        extender = list(np.zeros(shape=(len_orig - len_now)))
        trend = np.append(extender, trend)

        df["f_trend"] = trend
        df["f_trend.sig"] = ta.EMA(trend, 14)

        return df

    def big_trend(
        self, close: pd.Series, period: int, factor: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Identify the highest and lowest price during <period>
        highest = close.rolling(window=period).max()
        lowest = close.rolling(window=period).min()

        # Find the average of these values. For example, if the stock
        # reached a high of $591 and a low of $563, add them together
        # and divide the result by 2 to get an average of $577.
        average = (highest + lowest) / 2

        # Divide the difference between the high and low values by
        # their average. Continuing the example, divide $28 by $577
        # to get 0.0659.
        ratio = ((highest - lowest) / average) * 1000 * factor

        # Multiply this ratio by 2 and add 1. Continuing the example,
        # multiply 0.0659 by 2 and add 1 to get 1.132.
        ratio = ratio * 2 + 1

        # Multiply this factor by the stock's upper and lower values
        # to find the upper and lower acceleration bands. Continuing
        # the example, multiply $591 and $563 by 1.132 to get $669
        # and $637 as the stock's acceleration bands.
        upper_band = highest * ratio
        lower_band = lowest / ratio

        # return the average, upper and lower acceleration bands
        return (
            close.rolling(window=period).mean().to_numpy(),
            upper_band.rolling(window=period).mean().to_numpy(),
            lower_band.rolling(window=period).mean().to_numpy(),
        )

    # -------------------------------------------------------------------------
    def bollinger(
        self,
        data: TimeSeries,
        period: int = 20,
        multiplier: float = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mid = ta.SMA(data, timeperiod=period)
        std = ta.STDDEV(data, period)

        return mid, (mid + std * multiplier), (mid - std * multiplier)

    def keltner_channel(
        self,
        df: pd.DataFrame,
        period: int = 20,
        multiplier: float = 2,
        atr_lookback: int = 20,
    ) -> pd.DataFrame:
        atr = self.average_true_range(
            open_=df["open"],
            high_=df["high"],
            low_=df["low"],
            close_=df["close"],
            period=atr_lookback,
        )

        df["kc.mid"] = df["close"].ewm(span=period).mean()
        df["kc.upper"] = df["kc.mid"] + multiplier * atr
        df["kc.lower"] = df["kc.mid"] - multiplier * atr

        return df

    def keltner_talib(
        self,
        open_: TimeSeries,
        high: TimeSeries,
        low: TimeSeries,
        close: TimeSeries,
        period: int = 20,
        atr_multiplier: float = 2,
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """_summary_

        :param open_: open prices
        :type open_: TimeSeries
        :param high: high prices
        :type high: TimeSeries
        :param low: low prices
        :type low: TimeSeries
        :param close: close prices
        :type close: TimeSeries
        :param period: lookback for moving average/atr, defaults to 20
        :type period: int, optional
        :param atr_multiplier: _description_, defaults to 2
        :type atr_multiplier: float, optional
        :return: _description_
        :rtype: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        """

        for data in (open_, high, low, close):
            if isinstance(data, pd.Series):
                data = np.array(data.to_numpy(), dtype=np.float64)
            else:
                data = np.array(data, dtype=np.float64)

        kc_mid = np.array(ta.SMA(close, timeperiod=period))
        atr = ta.ATR(high, low, close, timeperiod=period)

        return tuple(
            (
                kc_mid,
                np.add(kc_mid, atr_multiplier * atr),
                np.subtract(kc_mid, atr_multiplier * atr),
            )
        )

    def cci(self, df: pd.DataFrame, period: int = 20):
        TP = (df["high"] + df["low"] + df["close"]) / 3
        df["cci"] = (TP - TP.rolling(period).mean()) / (
            0.015 * TP.rolling(period).std()
        )

        return df

    def adx(
        self, df: pd.DataFrame, lookback: int = 7, atr_lookback: int = 14
    ) -> pd.DataFrame:
        # calculate the ADR
        atr = self.average_true_range(
            open_=df.open,
            high_=df.high,
            low_=df.low,
            close_=df.close,
            period=atr_lookback,
        )

        # create a temporary dataframe for the calculations
        d = pd.DataFrame()
        d["high_diff"] = df.high.diff()
        d["low_diff"] = df.low.diff()

        d.loc[(d.high_diff > d.low_diff) & (d.high_diff > 0), "di_up"] = d.high_diff

        d.loc[(d.high_diff < d.low_diff) & (d.low_diff < 0), "di_down"] = d.high_diff

        # calculate all ADX components
        plus_di = 100 * (d.di_up.ewm(span=lookback).mean() / atr)
        minus_di = abs(100 * (d.di_down.ewm(span=lookback).mean() / atr))

        dx = abs(plus_di - minus_di) / abs(plus_di + minus_di)  # type: ignore
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = (adx.ewm(alpha=1 / lookback).mean()) * 100

        # assign columns and values to resulting dataframe
        df["adx.di.pls"] = plus_di
        df["adx.di.mns"] = minus_di
        df.loc[df["adx.di.pls"] > df["adx.di.mns"], "adx.trnd"] = 1
        df.loc[df["adx.di.pls"] < df["adx.di.mns"], "adx.trnd"] = -1
        df["adx"] = adx_smooth

        return df

    # -----------------------------------------------------------------------------
    #                                   MARKET STATE                              #
    # -----------------------------------------------------------------------------
    def choppiness_index(
        self, df: pd.DataFrame, lookback: int = 14, atr_lookback: int = 14
    ) -> pd.DataFrame:
        """Calculates the Choppiness Index for an asset.

        :param df: a Dataframe with columns 'open', 'high', 'low', 'close'
        :type df: pd.DataFrame
        :param lookback: lookback period for the CI, defaults to 14
        :type lookback: int, optional
        :param atr_lookback: lookback period for the ATR, defaults to 14
        :type atr_lookback: int, optional
        :return: a Dataframe with added column 'ci'
        :rtype: pd.DataFrame
        """
        atr = ta.ATR(df.high, df.low, df.close, atr_lookback)
        highest_high = df.high.rolling(lookback).max()
        lowest_low = df.low.rolling(lookback).min()

        df["ci"] = (
            100
            * np.log10((atr.rolling(lookback).sum()) / (highest_high - lowest_low))
            / np.log10(lookback)
        )

        return df

    def disparity_index(
        self,
        df: pd.DataFrame,
        column_a: pd.Series,
        column_b: pd.Series,
        column_c: pd.Series,
        as_percentiles: bool = False,
    ) -> pd.Series:
        # TODO: this is still incomplete?!
        return column_b / column_c - 1

    @execution_time
    def noise_index(
        self, data: pd.Series, period: int = 3, smoothing: int = 1
    ) -> pd.Series:
        returns = abs(data.pct_change())
        rolling_sum = returns.rolling(window=period).sum()
        price_diff = abs((data / data.shift(period)) - 1)

        return (price_diff / rolling_sum).rolling(window=smoothing).mean()

    @execution_time
    def trendy_index(self, data: pd.Series, lookback: int = 14) -> pd.Series:
        return (data.pct_change().fillna(0) * 100).rolling(window=lookback).sum()

    # -------------------------------------------------------------------------
    # helpers
    def _normalize_values(self, values: pd.Series, lookback: int):
        max = values.rolling(lookback).max()
        min = values.rolling(lookback).min()

        return ((values - min) / (max - min)) * 100

    def _percentiles(self, values: pd.Series, steps: int = 10) -> pd.Series:
        percentiles = [i / steps for i in range(steps)]
        return values.quantile([percentiles])

    def _column_exists(self, df: pd.DataFrame, column: str) -> bool:
        if column in df.columns:
            return True
        else:
            raise ValueError(f"{column} is not a valid column!")

    def _get_precision(self, column: pd.Series) -> int:
        string = str(column.iloc[1])

        if not "e" in string:
            try:
                prec = len(string.split(".")[-1])
            except Exception as e:
                print(e)
                print(column)
                prec = 8

        else:
            prec = 8

        return prec
