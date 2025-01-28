#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a statistics JIT class for often needed calculations.

Created on Tue Jan 28 17:33:23 2025

@author dhaneor
"""

from numba import float64
from numba.experimental import jitclass
import numpy as np

spec = [
    ('_epsilon', float64),
]

@jitclass(spec)
class Statistics:
    def __init__(self):
        self._epsilon = 1e-8  # Small value to avoid division by zero

    def _apply_to_columns(self, arr: np.ndarray, func, *args):
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
        arr_reshaped = arr.reshape(shape_0, shape_other)
        
        # Apply the function to each column
        for col in range(shape_other):
            column = arr_reshaped[:, col]
            arr_reshaped[:, col] = func(column, *args)
        
        # Reshape back to original shape
        return arr_reshaped.reshape(arr.shape)

    def _apply_rolling(self, arr: np.ndarray, func, period: int, *args):
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, arr.shape[0]):
            window = arr[i-period+1:i+1]
            result[i] = func(window, *args)
        return result

    def compute(self, arr: np.ndarray, func, period: int = 0, *args):
        if period == 0:
            return self._apply_to_columns(arr, func, *args)
        else:
            return self._apply_to_columns(arr, self._apply_rolling, func, period, *args)

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

    def cumsum(self, arr: np.ndarray) -> np.ndarray:
        return self.compute(arr, np.cumsum)

    def cumprod(self, arr: np.ndarray) -> np.ndarray:
        return self.compute(arr, np.cumprod)

    def diff(self, arr: np.ndarray, n: int = 1) -> np.ndarray:
        def diff_func(x, n):
            return np.diff(x, n)
        return self.compute(arr, diff_func, 0, n)

    def pct_change(self, arr: np.ndarray, period: int = 1) -> np.ndarray:
        def pct_change_func(x, period):
            return (x[period:] - x[:-period]) / (x[:-period] + self._epsilon)
        return self.compute(arr, pct_change_func, 0, period)

    def log_returns(self, arr: np.ndarray) -> np.ndarray:
        def log_return_func(x):
            return np.log(x[1:] / x[:-1])
        return self.compute(arr, log_return_func)

    def annualized_returns(self, arr: np.ndarray, interval_ms: int) -> np.ndarray:
        def annualized_return_func(x, interval_ms):
            total_return = x[-1] / x[0] - 1
            years = (len(x) * interval_ms) / self._ms_per_year
            return (1 + total_return) ** (1 / years) - 1
        return self.compute(arr, annualized_return_func, 0, interval_ms)

    def annualized_volatility(self, arr: np.ndarray, interval_ms: int) -> np.ndarray:
        def annualized_vol_func(x, interval_ms):
            log_returns = np.log(x[1:] / x[:-1])
            periods_per_year = self._ms_per_year / interval_ms
            return np.std(log_returns) * np.sqrt(periods_per_year)
        return self.compute(arr, annualized_vol_func, 0, interval_ms)

    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float, interval_ms: int) -> np.ndarray:
        def sharpe_ratio_func(x, risk_free_rate, interval_ms):
            periods_per_year = self._ms_per_year / interval_ms
            excess_returns = x - (risk_free_rate / periods_per_year)
            return np.mean(excess_returns) / (np.std(excess_returns) + self._epsilon) * np.sqrt(periods_per_year)
        return self.compute(returns, sharpe_ratio_func, 0, risk_free_rate, interval_ms)

    def true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        def tr_func(h, l, c):
            return np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
        return self.compute(np.stack((high, low, close), axis=-1), tr_func)

    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        tr = self.true_range(high, low, close)
        return self.mean(tr, period)

    def max_drawdown(self, arr: np.ndarray) -> np.ndarray:
        def mdd_func(x):
            peak = np.maximum.accumulate(x)
            drawdown = (x - peak) / peak
            return np.min(drawdown)
        return self.compute(arr, mdd_func)

    def rolling_max_drawdown(self, arr: np.ndarray, period: int) -> np.ndarray:
        def rolling_mdd_func(window):
            peak = np.maximum.accumulate(window)
            drawdown = (window - peak) / peak
            return np.min(drawdown)
        return self.compute(arr, rolling_mdd_func, period)