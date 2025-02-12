#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a statistics JIT class for often needed calculations.

Created on Tue Jan 28 17:33:23 2025

@author dhaneor
"""

from numba import float64, int64
from numba.experimental import jitclass
import numpy as np


spec = [
    ("_epsilon", float64),
    ("periods_per_year", int64),
    ("risk_free_rate", float64),
]


@jitclass(spec)
class Statistics:
    def __init__(self):
        self._epsilon = 1e-8  # Small value to avoid division by zero
        self.periods_per_year = 365
        self.risk_free_rate = 0.01  # 1% riskfree rate

    def _apply_to_columns(self, arr: np.ndarray, func, period: int) -> np.ndarray:
        ndim = arr.ndim
        if ndim < 1:
            raise ValueError("Input array must have at least 1 dimension")

        # Calculate the total number of columns across all other dimensions
        num_cols = 1
        for dim in range(1, ndim):
            num_cols *= arr.shape[dim]

        # Reshape the array to 2D: (axis=0, all other axes flattened)
        shape_0 = arr.shape[0]
        shape_other = num_cols

        if period > 0:
            arr_reshaped = arr.reshape(shape_0, shape_other)

            # Apply the function to each column
            for col in range(shape_other):
                column = arr_reshaped[:, col]
                arr_reshaped[:, col] = self._apply_rolling(column, func, period)
            # Reshape back to original shape
            return arr_reshaped.reshape(arr.shape)

        else:
            # create an empty array with the same shape as the input array,
            # except for first dimension, which now has a length of 1
            out = np.full((1, shape_other), np.nan, dtype=np.float64)
            arr_reshaped = arr.reshape(shape_0, shape_other)

            for col in range(shape_other):
                column = arr_reshaped[:, col]
                out[0, col] = func(column)

            new_shape = (1,) + arr.shape[1:]
            return out.reshape(new_shape)

    def _apply_rolling(self, arr: np.ndarray, func, period: int) -> np.ndarray:
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, arr.shape[0]):
            window = arr[i - period + 1 : i + 1]
            result[i] = func(window)
        return result

    def compute(self, arr: np.ndarray, func, period: int) -> np.ndarray:
        return self._apply_to_columns(arr, func, period)

    # ................................. Simple Stats ...................................
    def mean(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.mean, period)

    def std(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.std, period)

    def var(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.var, period)

    def min(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.min, period)

    def max(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.max, period)

    def sum(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.sum, period)

    # ................................. Returns and Volatility .........................
    def returns(self, arr: np.ndarray, period: int = 0):
        return np.diff(arr) / arr[:-1]

    def log_returns(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, self._log_returns_fn, period)

    def volatility(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, np.std, period)

    def sharpe_ratio(self, arr: np.ndarray, period: int = 0) -> np.ndarray:
        return self.compute(arr, self._sharpe_ratio_fn, period)

    # ......................... Annualized Returns and Volatility ......................
    def annualized_returns(
        self, arr: np.ndarray, periods_per_year: int, period: int = 0
    ) -> np.ndarray:
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_returns_fn, period)

    def annualized_volatility(
        self, arr: np.ndarray, periods_per_year: int, period: int = 0
    ) -> np.ndarray:
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_vol_fn, period)

    def annualized_sharpe_ratio(
        self, arr: np.ndarray, periods_per_year: int, period: int = 0
    ) -> np.ndarray:
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_sharpe_ratio_fn, period)

    # ................................. Helper methods .................................
    def _returns_fn(self, arr: np.ndarray) -> np.ndarray:
        return np.diff(arr, n=1, axis=0)

    def _log_returns_fn(self, arr: np.ndarray) -> np.ndarray:
        return np.log(np.diff(arr))

    def _sharpe_ratio_fn(self, arr: np.ndarray) -> np.ndarray:
        # Calculate periodic returns
        n = arr.shape[0]
        returns = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            returns[i] = (arr[i + 1] / arr[i]) - 1

        mean_return = np.sum(returns) / len(returns)
        std_dev = np.std(returns)

        return mean_return / std_dev

    def _annualized_returns_fn(self, arr: np.ndarray) -> np.ndarray:
        total_return = arr[-1] / arr[0] - 1
        return (1 + total_return) ** (self.periods_per_year / len(arr)) - 1

    def _annualized_vol_fn(self, arr: np.ndarray):
        log_returns = np.log(arr[1:] / arr[:-1])
        return np.std(log_returns) * np.sqrt(self.periods_per_year)

    def _annualized_sharpe_ratio_fn(self, arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2:
            return 0.0

        # Calculate periodic returns
        returns = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            returns[i] = (arr[i + 1] / arr[i]) - 1

        mean_return = np.sum(returns) / len(returns)
        annualized_return = mean_return * self.periods_per_year

        # Calculate standard deviation
        sum_sq_diff = 0.0
        for r in returns:
            diff = r - mean_return
            sum_sq_diff += diff * diff
        std_dev = np.sqrt(sum_sq_diff / (len(returns) - 1))

        # Annualized volatility
        annualized_volatility = std_dev * np.sqrt(self.periods_per_year)

        # Excess return
        excess_return = annualized_return - self.risk_free_rate

        # Sharpe ratio
        sharpe_ratio = excess_return / (annualized_volatility + self._epsilon)

        return sharpe_ratio

    # ...............................ATR calculation .................................
    def true_range(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        def tr_func(h, l, c):
            return np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))

        return self.compute(np.stack((high, low, close), axis=-1), tr_func)

    def atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        tr = self.true_range(high, low, close)
        return self.mean(tr, period)
