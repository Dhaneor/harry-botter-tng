#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 28 20:08:20 2023

@author dhaneor
"""
import bottleneck as bn
import numpy as np

from math import exp, cos, pi
from numba import njit
from . import IIndicator, Parameter


class EfficiencyRatio(IIndicator):
    """A class to calculate the Efficiency Ratio (ER) of a given time series.

    The Efficiency Ratio is a technical indicator used to measure
    the noise in the market. It is determined by calculating the
    ratio between:
    a) the absolute price difference between the current price
    and the previous price at the beginning of the lookback period
    b) the sum of the price differences for all lookback periods
    
    """
    method: int = 1  # calculation method (0=numpy, 1=numba)

    def __init__(self):
        super().__init__()
        self._name = "er"
        self.display_name = "Efficiency Ratio"

        self.input = ["close"]
        self.output = ["er"]
        self.output_flags = dict(real=["Line"])

        self._parameters = (
            Parameter(
                name="timeperiod", 
                initial_value=14, 
                hard_min=5, 
                hard_max=200, 
                step=5
            ),
            Parameter(
                name="smoothing", 
                initial_value=1, 
                hard_min=1, 
                hard_max=15, 
                step=3
            ),
        )
        self._plot_desc = dict(real=["Line"])

    def _apply_func(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the (Kaufman) Efficiency Ratio.

        The Efficiency Ratio is a technical indicator used to measure
        the noise in the market. It is determined by calculating the
        ratio between:
        a) the absolute price difference between the current price
        and the previous price at the beginning of the lookback period
        b) the sum of the price differences for all lookback periods


        Parameters
        ----------
        data : np.ndarray
            A one- or two-dimensional array of price data.

        Returns
        -------
        np.ndarray
            The Noise indicator values for the lookback set
            in the 'timeperiod' parameter (default: 14).
        """
        smoothing = self.parameters[1].value

        if self._cache is None:
            if self.method == 1:
                res = noise_index_nb(
                    data=data, 
                    lookback=self.parameters[0].value
                    )
            else:
                res = self._noise_index_numpy(data)


            if smoothing > 1:
                res = ultimate_smoother(
                    np.nan_to_num(res), 
                    smoothing
                )

            self._cache = res
        
        return self._cache

    def _noise_index_numpy(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Noise index using NumPy operations.

        This method computes the Noise index, which measures the ratio
        of the net price movement to the sum of absolute price
        movements over a specified lookback period.

        Parameters:
        -----------
        data : np.ndarray
            A 1-dimensional numpy array of price data.

        Returns:
        --------
        np.ndarray
            An array of Noise index values, with the same length as the
            input data. The first 'lookback' elements are padded with
            NaN values.

        Notes:
        ------
        The Noise index is calculated as:
        (abs(final_price / initial_price - 1)) / (sum of absolute returns)
        """
        lookback = min((len(data) - 1), self._parameters[0].value)

        returns = np.abs(np.diff(data) / data[:-1])
        rolling_sum = bn.move_sum(returns, window=lookback)

        price_diff = np.abs(data[lookback:] / data[:-lookback] - 1)

        noise = price_diff / rolling_sum[lookback - 1:]

        # Pad the beginning of the array with NaNs to match the input length
        return np.pad(
            noise, (len(data) - len(noise), 0), mode="constant", constant_values=np.nan
        )


@njit
def noise_index_nb(data, lookback: int) -> np.ndarray:
    n = len(data)
    noise = np.empty(n)
    noise[:] = np.nan

    for i in range(lookback, n):
        returns = np.abs(
            np.diff(data[i - lookback: i]) / data[i - lookback: i - 1]
        )
        rolling_sum = np.sum(returns)
        price_diff = np.abs(data[i] / data[i - lookback] - 1)
        noise[i] = price_diff / rolling_sum

    return noise


@njit
def efficiency_ratio_nb(data, lookback: int) -> np.ndarray:
    n = len(data)
    er = np.empty(n)
    er[:] = np.nan

    for i in range(lookback, n):
        # Absolute price change over the lookback period
        price_diff = np.abs(data[i] - data[i - lookback])

        # Sum of absolute price changes within the lookback window
        window_start = i - lookback
        window_end = i
        sum_abs_changes = 0.0
        for j in range(window_start + 1, window_end):
            sum_abs_changes += np.abs(data[j] - data[j - 1])

        # Calculate Efficiency Ratio
        if sum_abs_changes != 0.0:
            er[i] = price_diff / sum_abs_changes
        else:
            er[i] = np.nan

    return er


@njit
def noise_index_weighted_nb(data, lookback: int) -> np.ndarray:
    n = len(data)
    noise = np.empty(n)
    noise[:] = np.nan

    alpha = 2.0 / (lookback + 1)
    
    for i in range(lookback, n):
        # Calculate returns in the lookback window
        window_data = data[i - lookback: i]
        returns = np.abs(np.diff(window_data) / window_data[:-1])

        # Build weights (most recent has largest weight)
        weights = np.array([alpha * (1 - alpha)**k for k in range(lookback - 1, -1, -1)])
        # If desired, normalize so sum of weights == 1
        weights /= np.sum(weights)
        
        # Weighted sum of returns
        weighted_sum = np.sum(returns * weights[1:])  # returns is length (lookback-1)
        
        # "Price diff" can stay the same or be adjusted with the first weight
        price_diff = np.abs(data[i] / data[i - lookback] - 1)
        
        noise[i] = price_diff / weighted_sum if weighted_sum != 0 else np.nan

    return noise



@njit
def weighted_efficiency_ratio_nb(data, lookback: int) -> np.ndarray:
    n = len(data)
    er = np.empty(n)
    er[:] = np.nan

    alpha = 2.0 / (lookback + 1)

    # Precompute weights: weights[k] corresponds to data[i - lookback + k]
    weights = np.empty(lookback - 1)
    for k in range(lookback - 1):
        weights[k] = alpha * (1 - alpha) ** (lookback - 2 - k)  # Exponential weights

    # Normalize weights so that sum(weights) = 1
    weight_sum = 0.0
    for k in range(lookback - 1):
        weight_sum += weights[k]
    for k in range(lookback - 1):
        weights[k] /= weight_sum if weight_sum != 0 else 1.0

    for i in range(lookback, n):
        # Absolute price change over the lookback period
        price_diff = np.abs(data[i] - data[i - lookback])

        # Sum of weighted absolute price changes within the lookback window
        window_start = i - lookback
        window_end = i
        weighted_sum = 0.0
        for j in range(window_start + 1, window_end):
            weighted_sum += weights[j - (window_start + 1)] * np.abs(data[j] - data[j - 1])

        # Calculate Weighted Efficiency Ratio
        if weighted_sum != 0.0:
            er[i] = price_diff / weighted_sum
        else:
            er[i] = np.nan

    return er



@njit
def noise_index_weighted_alt_nb(data, lookback: int) -> np.ndarray:
    n = len(data)
    noise = np.empty(n)
    noise[:] = np.nan

    alpha = 2.0 / (lookback + 1)

    # Precompute weights: weights[k] corresponds to data[i - lookback + k]
    weights = np.empty(lookback)
    for k in range(lookback):
        weights[k] = alpha * (1 - alpha) ** (lookback - 1 - k)
    
    # Normalize weights so that sum(weights[:-1]) == 1 for the returns
    # Since returns have (lookback - 1) elements
    weight_sum = 0.0
    for k in range(lookback - 1):
        weight_sum += weights[k + 1]
    for k in range(lookback):
        weights[k] /= weight_sum if weight_sum != 0 else 1.0

    # Loop through each data point starting from 'lookback'
    for i in range(lookback, n):
        # Calculate returns in the lookback window
        # data[i - lookback : i] has 'lookback' points
        # returns will have (lookback - 1) elements
        window_start = i - lookback
        window_end = i
        # Compute absolute returns
        returns = np.empty(lookback - 1)
        for j in range(lookback - 1):
            returns[j] = np.abs((data[window_start + j + 1] - data[window_start + j]) / data[window_start + j])
        
        # Compute weighted sum of returns
        weighted_sum = 0.0
        for j in range(lookback - 1):
            weighted_sum += returns[j] * weights[j + 1]
        
        # Compute price difference
        price_diff = np.abs(data[i] / data[i - lookback] - 1)
        
        # Calculate noise index
        if weighted_sum != 0.0:
            noise[i] = price_diff / weighted_sum
        else:
            noise[i] = np.nan

    return noise


# ======================================================================================
class UltimateSmoother(IIndicator):
    """A class to calculate the Ultimate Smoother.

    The Ultimate Smoother is a second-order recursive  low-pass 
    filter designed to smooth out data (such  as price series in 
    financial markets) while minimizing lag and preserving trend 
    information. Its recursive nature means that each smoothed 
    value depends on both current and past data points as well as 
    past smoothed values.
    """

    def __init__(self):
        super().__init__()
        self._name = "us"
        self.display_name = "Ultimate Smoother"

        self.input = ["close"]
        self.output = ["us"]
        self.output_flags = dict(real=["Line"])

        self._parameters = (
            Parameter(
                name="timeperiod", 
                initial_value=14, 
                hard_min=2, 
                hard_max=100, 
                step=5
            ),
        )
        self._plot_desc = dict(real=["Line"])
        self._is_subplot = False

    def _apply_func(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates Ehler#s Ultmiate Smmother

        The Ultimate Smoother is a second-order recursive  low-pass 
        filter designed to smooth out data (such  as price series in 
        financial markets) while minimizing lag and preserving trend 
        information. Its recursive nature means that each smoothed 
        value depends on both current and past data points as well as 
        past smoothed values.

        Parameters
        ----------
        data : np.ndarray
            A one- or two-dimensional array of price data.
        timeperiod : int, optional
            The smoothing period (default: 14).

        Returns
        -------
        np.ndarray
            The Ultimate Smoother values for the lookback set
            in the 'timeperiod' parameter (default: 14).
        """
        
        if self._cache is None:
            self._cache = ultimate_smoother(data, self.parameters[0].value)
        
        return self._cache
    

@njit
def ultimate_smoother(data, length: np.int64 = 20) -> np.ndarray:
    """
    Calculate the Ehlers Ultimate Smoother for a given data series.

    Parameters:
    - data (array-like): The input data series (e.g., prices).
    - length (int): The smoothing period.

    Returns:
    - us (np.ndarray): The smoothed data series.
    """
    # length = self.parameters[0].value
    n: np.int64 = len(data)
    us: np.ndarray = np.zeros(n, dtype=np.float64)
    
    # Check if data length is sufficient
    if n < 3:
        raise ValueError("Data array must have at least 3 elements.")
    
    # Initialize the smoothed series with zeros
    us = np.zeros(n)

    f: np.float64 = (1.414 * pi) / length
    a1 = exp(-f)
    c2 = 2 * a1 * cos(f)
    
    # # Calculate the frequency component
    # f = (1.414 * np.pi) / length
    
    # # Compute intermediate coefficients
    # a1 = np.exp(-f)
    # c2 = 2 * a1 * np.cos(f)
    c3 = -a1 ** 2
    c1 = (1 + c2 - c3) / 4
    
    # Initialization:
    us[0] = data[0]
    us[1] = data[1]
    
    # Iterate through the data starting from index 2
    for t in range(2, n):
        us[t] = (
            (1 - c1) * data[t] +
            (2 * c1 - c2) * data[t - 1] +
            (-c1 - c3) * data[t - 2] +
            c2 * us[t - 1] +
            c3 * us[t - 2]
        )
    
    return us


# ======================================================================================
custom_indicators = {
    "ER": EfficiencyRatio,
    "US": UltimateSmoother,
    }
