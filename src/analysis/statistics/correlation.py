#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import numpy as np
import time
from numba import njit, prange, int8  # noqa: F401, E402
from numba.experimental import jitclass

init_data = np.random.rand(100, 10)


@njit(cache=True)  # Cache the compiled function for better performance.
def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate correlations for an n-dimensional numpy array.

    Parameters
    ----------
    data : np.ndarray
        An n-dimensional numpy array where each column represents a different asset.

    Returns
    -------
    np.ndarray
        A 2D correlation matrix.
    """
    correlations = np.corrcoef(data, rowvar=False)
    np.fill_diagonal(correlations, np.nan)
    return correlations


def rolling_mean_correlation_python(arr: np.ndarray, period: int) -> np.ndarray:
    """Returns the rolling mean correlation between the close prices.

    NOTE: This is the first 'naive' implementation, but it cannot 
    be accelearated with Numba, because Numba does not like the
    np.triuindices method. Just leave this here for reference and
    to be able to check the correctness of the results of the 
    final algorithm.

    Parameters
    ----------
    arr: np.ndarray
        A 2D numpy array representing prices or euqity curves, one
        column per asset or strategy

    Returns
    -------
    np.ndarray
        A 1D numpy array of the rolling mean correlations.
    """
    num_assets = arr.shape[1]
    num_periods = arr.shape[0]

    rolling_mean_corr = np.full(num_periods, np.nan)

    for i in range(period - 1, num_periods):
        window = arr[i - period + 1 : i + 1]
        correlations = correlation_matrix(window)      
        rolling_mean_corr[i] = np.mean(
            correlations[np.triu_indices(num_assets, k=1)]
        )

    return rolling_mean_corr


@njit(parallel=True, cache=True)  # Cache the compiled function for better performance.
def rolling_mean_correlation_nb_old(arr: np.ndarray, period: int) -> np.ndarray:
    """Returns the rolling mean correlation between the close prices.

    NOTE: This was the second version of the algorithm which is compatible
    with Numba and uses parallel execution. It is still slower than the 
    final version (below).

    Parameters
    ----------
    arr: np.ndarray
        A 2D numpy array representing prices or equity curves, one
        column per asset or strategy
    period: int
        The rolling window size

    Returns
    -------
    np.ndarray
        A 1D numpy array of the rolling mean correlations.
    """
    num_periods, num_assets = arr.shape
    rolling_mean_corr = np.full(num_periods, np.nan)

    for i in prange(period - 1, num_periods):
        window = arr[i - period + 1 : i + 1]
        correlations = correlation_matrix(window)
        
        # Calculate mean of all values, excluding the diagonal
        total = 0.0
        count = 0
        for j in range(num_assets):
            for k in range(j + 1, num_assets):
                total += correlations[j, k]
                count += 1
        
        rolling_mean_corr[i] = total / count if count > 0 else np.nan

    return rolling_mean_corr


# ======================================================================================
@njit(parallel=True, cache=True)
def rolling_mean_correlation(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Optimized calculation of rolling mean correlation for multiple assets.

    Parameters
    ----------
    arr : np.ndarray
        A 2D numpy array of shape (n_periods, n_assets), where each 
        column represents an asse/strategy.
    period : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        A 1D numpy array of rolling mean correlations.
    """

    n_periods, n_assets = arr.shape
    rolling_mean_corr = np.full(n_periods, np.nan, dtype=np.float64)

    # Initialize rolling sums and variances
    rolling_sums = np.zeros((n_periods, n_assets))
    rolling_sq_sums = np.zeros((n_periods, n_assets))
    
    # Compute rolling sums and squared sums
    for i in prange(n_assets):
        for t in range(n_periods):
            if t >= period - 1:
                rolling_sums[t, i] = np.sum(arr[t - period + 1 : t + 1, i])
                rolling_sq_sums[t, i] = np.sum(arr[t - period + 1 : t + 1, i] ** 2)

    # Calculate rolling correlations
    for t in prange(period - 1, n_periods):
        means = rolling_sums[t] / period
        variances = (rolling_sq_sums[t] / period) - means**2

        correlations_sum = 0.0
        count = 0
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                cov = np.mean((arr[t - period + 1 : t + 1, i] - means[i]) * 
                              (arr[t - period + 1 : t + 1, j] - means[j]))
                corr = cov / (np.sqrt(variances[i]) * np.sqrt(variances[j]) + 1e-10)
                correlations_sum += corr
                count += 1

        rolling_mean_corr[t] = correlations_sum / count if count > 0 else np.nan

    return rolling_mean_corr


@njit(cache=True, parallel=True)
def mean_correlation(arr: np.ndarray) -> float:
    """
    Calculate the mean correlation between all unique pairs of assets in the given lookback window.

    Parameters
    ----------
    arr : np.ndarray
        A 2D numpy array of shape (n_periods, n_assets) where each column represents an asset/strategy.

    Returns
    -------
    float
        The mean correlation between all unique asset pairs. If fewer than two assets are provided,
        returns np.nan.
    """
    n_periods, n_assets = arr.shape

    # If there are fewer than 2 assets, we cannot compute pairwise correlations.
    if n_assets < 2:
        return np.nan

    # Calculate means for each asset.
    means = np.empty(n_assets, dtype=arr.dtype)
    for i in prange(n_assets):
        s = 0.0
        for t in range(n_periods):
            s += arr[t, i]
        means[i] = s / n_periods

    # Calculate variances for each asset.
    variances = np.empty(n_assets, dtype=arr.dtype)
    for i in range(n_assets):
        var = 0.0
        for t in range(n_periods):
            diff = arr[t, i] - means[i]
            var += diff * diff
        variances[i] = var / n_periods

    # Calculate pairwise correlations.
    correlations_sum = 0.0
    count = 0
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            cov = 0.0
            for t in range(n_periods):
                cov += (arr[t, i] - means[i]) * (arr[t, j] - means[j])
            cov = cov / n_periods
            # Adding a small epsilon to avoid division by zero
            corr = cov / ((np.sqrt(variances[i]) * np.sqrt(variances[j])) + 1e-10)
            correlations_sum += corr
            count += 1

    if count == 0:
        return np.nan
    else:
        return correlations_sum / count


@njit(cache=True, parallel=True)
def levels_to_log_returns(arr: np.ndarray) -> np.ndarray:
    n_periods, n_assets = arr.shape
    returns = np.empty((n_periods - 1, n_assets), dtype=arr.dtype)
    for i in prange(n_assets):
        for t in range(1, n_periods):
            prev = arr[t - 1, i]
            # Ensure the previous price is positive to avoid taking log(0) or log of a negative number.
            if prev <= 0:
                returns[t - 1, i] = 0.0
            else:
                returns[t - 1, i] = np.log(arr[t, i] / prev)
    return returns



@jitclass(spec=[("period", int8)])
class Correlation:
    """A class for correlation-related computations.
    
    Clients should decide if they want to provide levels or returns. For
    most uses cases, returns are preferrable.

    •	Normalization Across Assets:
    Percentage returns (or rate of change) normalize the data so that 
    you’re comparing relative performance rather than absolute price 
    movements. This way, a move from $10 to $11 is treated similarly to 
    a move from $100 to $110, which is essential when asset prices 
    differ significantly.
    
    •	Stationarity:
    Percentage returns tend to be more stationary compared to raw price 
    levels. Stationary data are generally better for statistical analysis, 
    including correlation calculations, because the underlying properties 
    (mean, variance) remain more consistent over time.
    
    •	Risk Assessment:
    Risk and diversification analysis (like calculating correlations) is 
    more meaningful on returns. By using percentage returns, you ensure 
    that the risk assessments reflect the actual volatility and 
    co-movement between the assets, independent of their scale.
    
    •	Alternative Consideration – Log Returns:
    Many practitioners and researchers also use log returns because they 
    have the desirable property of time-additivity (i.e., the log return
    over multiple periods is the sum of the log returns over each period). 
    However, for most practical backtesting and risk management purposes,
    percentage returns (simple returns) are sufficiently robust and easier 
    to interpret.
    """

    def __init__(self, period: int = 20):
        self.period = period

    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Calculate correlations for an n-dimensional numpy array.

        Parameters
        ----------
        data : np.ndarray
            An n-dimensional numpy array where each column represents 
            a different asset.

        Returns
        -------
        np.ndarray
            A 2D correlation matrix.
        """
        correlations = np.corrcoef(data, rowvar=False)
        np.fill_diagonal(correlations, np.nan)
        return correlations

    def mean(self, arr: np.ndarray) -> float:
        """Calcualtes the mean correlation of 2-dimensional numpy array
        
        Parameters:
        ----------
        arr : np.ndarray
            A 2D numpy array where each column represents  the _returns_
            of a different asset/strategy.

        Returns:
        --------
        float
            The mean correlation as a scalar value.
        
        """
        return mean_correlation(arr)

    def rolling_mean(self, arr: np.ndarray):
        """
        Optimized calculation of rolling mean correlation for multiple assets.

        Note: I moved the function out of the class beaucse then it is 
        possible to use the 'parallel' argmument, which makes the 
        calculation about 15x faster.

        Parameters
        ----------
        arr : np.ndarray
            A 2D numpy array of shape (n_periods, n_assets), where each column 
            represents the _returns_ of an asset/strategy.
        period : int
            Rolling window size.

        Returns
        -------
        np.ndarray
            A 1D numpy array of rolling mean correlations.
        """
        return rolling_mean_correlation(arr, self.period)

    # ...........Helper methods for clients to transform prices to returns.............
    def levels_to_pct_returns(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert price levels to percentage returns.

        Parameters
        ----------
        arr : np.ndarray
            A 2D numpy array of shape (n_periods, n_assets) representing price levels.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (n_periods-1, n_assets) with percentage returns.
            The return is computed as: (current - previous) / previous.
        """
        n_periods, n_assets = arr.shape
        # The returns array will have one fewer row than the levels array.
        returns = np.empty((n_periods - 1, n_assets), dtype=arr.dtype)
        for i in range(n_assets):
            for t in range(1, n_periods):
                prev = arr[t - 1, i]
                # Avoid division by zero.
                if prev == 0:
                    returns[t - 1, i] = 0.0
                else:
                    returns[t - 1, i] = (arr[t, i] - prev) / prev
        return returns

    def levels_to_log_returns(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert price levels to log returns.

        Parameters
        ----------
        arr : np.ndarray
            A 2D numpy array of shape (n_periods, n_assets) representing price levels.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (n_periods-1, n_assets) with log returns.
            The return is computed as: log(current / previous).
        """
        return levels_to_log_returns(arr)

        n_periods, n_assets = arr.shape
        returns = np.empty((n_periods - 1, n_assets), dtype=arr.dtype)
        for i in range(n_assets):
            for t in range(1, n_periods):
                prev = arr[t - 1, i]
                # Ensure the previous price is positive to avoid taking log(0) or log of a negative number.
                if prev <= 0:
                    returns[t - 1, i] = 0.0
                else:
                    returns[t - 1, i] = np.log(arr[t, i] / prev)
        return returns


# ======================================================================================
def test():
    correlation = Correlation(21)

    # Example usage
    arr = np.random.rand(1000, 20)  # (periods, assets)
    arr_short = np.random.rand(21, 20)  # (periods, assets)

    # initialize the Numba functions/class
    correlation.mean(arr)
    correlation.levels_to_log_returns(arr) 
    correlation.levels_to_pct_returns(arr)
    correlation.rolling_mean(arr)
    rolling_mean_correlation(arr, 21)
    levels_to_log_returns(arr_short)
    mean_correlation(arr_short)

    st = time.time()

    for _ in range(1000):
        # res = levels_to_log_returns(arr_short)
        # res = correlation.levels_to_pct_returns(arr_short)
        # res = correlation.mean(arr_short)
        # res = correlation.rolling_mean(arr)
        res = rolling_mean_correlation(arr, 21)

    et = time.time() - st

    print(F"avg exc time: {et / 100 * 1e6:.2f}µs")

    if isinstance(res, float):
        print(res)
    else:
        print(res[-10:])
        print("result dtype: ", res.dtype)

if __name__ == "__main__":
    test()