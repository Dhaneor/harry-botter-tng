#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import numpy as np
from numba import njit  # noqa: F401, E402

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

correlation_matrix(init_data)


@njit(cache=True)  # Cache the compiled function for better performance.
def rolling_mean_correlation(arr: np.ndarray, period: int) -> np.ndarray:
    """Returns the rolling mean correlation between the close prices.

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
        rolling_mean_corr[i] = np.nanmean(
            correlations[np.triu_indices(num_assets, k=1)]
        )

    return rolling_mean_corr


class Correlation:

    def matrix(self, data: np.ndarray) -> np.ndarray:
        return correlation_matrix(data)

    def mean(self, data: np.ndarray) -> np.ndarray:
        """Returns the mean correlation between the close prices."""
        correlations = self.matrix(data)
        return np.nanmean(
            correlations[np.triu_indices(correlations.shape[0], k=1)]
            )

    def rolling(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Returns the rolling mean correlation between the close prices."""
        num_assets = arr.shape[1]
        num_periods = arr.shape[0]

        rolling_mean_corr = np.full(num_periods, np.nan)

        for i in range(period - 1, num_periods):
            window = arr[i - period + 1 : i + 1]
            correlations = correlation_matrix(window)
            rolling_mean_corr[i] = np.nanmean(
                correlations[np.triu_indices(num_assets, k=1)]
            )

        return rolling_mean_corr


def test():
    correlation = Correlation()

    # Example usage
    arr = np.random.rand(100_000, 10)  # (periods, assets)
    # print(correlation.rolling(arr, period=20))
    # print(correlation.matrix(arr))
    # print(correlation.matrix(arr))

    correlation.rolling(arr, period=20)
    correlation.matrix(arr)

if __name__ == "__main__":
    test()