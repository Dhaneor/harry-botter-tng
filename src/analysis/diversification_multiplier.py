#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 04:50:23 2025

@author: dhaneor
"""
import numpy as np
from numba import njit, int64, float64, float32
from numba.experimental import jitclass

from analysis.statistics.correlation import Correlation


# These arrays represent the DM_MATRIX keys and values.
# Outer keys (correlation keys):
CORR_KEYS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

# Inner keys (number-of-assets keys):
ASSETS_KEYS = np.array([2, 3, 4, 5, 10, 15, 20, 50], dtype=np.int32)

# The multiplier values (rows correspond to corr_keys, columns to assets_keys).
DM_VALUES = np.array([
    [1.41, 1.73, 2.0,  2.2,  3.2,  3.9,  4.5,  7.1],
    [1.27, 1.41, 1.51, 1.58, 1.75, 1.83, 1.86, 1.94],
    [1.15, 1.22, 1.27, 1.29, 1.35, 1.37, 1.38, 1.40],
    [1.10, 1.12, 1.13, 1.15, 1.17, 1.17, 1.18, 1.19],
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]
], dtype=np.float32)


@njit(cache=True)
def calculate_multiplier_single(mean_corr: float, num_assets: int):
    # First, determine the closest asset count key.
    assets_keys = ASSETS_KEYS
    corr_keys = CORR_KEYS

    if num_assets > 50:
        num_assets = 50  # as in your original code, use min(self._data.shape[1], 50)
    
    closest_asset = assets_keys[0]
    min_diff = abs(assets_keys[0] - num_assets)
    for i in range(1, assets_keys.shape[0]):
        diff = abs(assets_keys[i] - num_assets)
        if diff < min_diff:
            # print("min_diff =", min_diff)
            min_diff = diff
            closest_asset = assets_keys[i]
            # print("closest: ", closest_asset)
            # print("-" * 50)
    
    # Find the index for the closest asset key.
    asset_index = 0
    for i in range(assets_keys.shape[0]):
        if assets_keys[i] == closest_asset:
            asset_index = i
            break

    corr = mean_corr * num_assets
    # Find the closest correlation key.
    if corr <= corr_keys[0]:
        corr_index = 0
    elif corr >= corr_keys[-1]:
        corr_index = corr_keys.shape[0] - 1
    else:
        # Loop through the keys to find where corr fits.
        corr_index = 0
        for j in range(1, corr_keys.shape[0]):
            if corr < corr_keys[j]:
                # Check which key is closer: the one before or this one.
                if abs(corr - corr_keys[j - 1]) <= abs(corr_keys[j] - corr):
                    corr_index = j - 1
                else:
                    corr_index = j
                break
    return DM_VALUES[corr_index, asset_index]


@njit(cache=True)
def calculate_multiplier(rolling_mean_corr: np.ndarray, num_assets: int):
    """
    Computes the diversification multiplier for each time step.
    
    Parameters:
      rolling_mean_corr : 2D np.ndarray of shape (n, 1)
          Rolling mean correlation values.
      num_assets : int
          The number of assets (after applying the min(num_assets, 50) rule).
      corr_keys : 1D np.ndarray
          Array of correlation keys.
      assets_keys : 1D np.ndarray
          Array of asset keys.
      dm_values : 2D np.ndarray
          2D array with multiplier values; rows correspond to corr_keys,
          columns correspond to assets_keys.
    
    Returns:
      result : 2D np.ndarray of shape (n, 1)
          The diversification multipliers.
    """

    corr_keys = CORR_KEYS
    assets_keys = ASSETS_KEYS
    dm_values = DM_VALUES

    n = rolling_mean_corr.shape[0]
    result = np.ones((n, 1), dtype=dm_values.dtype)
    
    # First, determine the closest asset count key.
    if num_assets > 50:
        num_assets = 50  # as in your original code, use min(self._data.shape[1], 50)
    
    closest_asset = assets_keys[0]
    min_diff = abs(assets_keys[0] - num_assets)
    for i in range(1, assets_keys.shape[0]):
        diff = abs(assets_keys[i] - num_assets)
        if diff < min_diff:
            # print("min_diff =", min_diff)
            min_diff = diff
            closest_asset = assets_keys[i]
            # print("closest: ", closest_asset)
            # print("-" * 50)
    
    # Find the index for the closest asset key.
    asset_index = 0
    for i in range(assets_keys.shape[0]):
        if assets_keys[i] == closest_asset:
            asset_index = i
            break

    # print("closest number of assets: ", ASSETS_KEYS[asset_index])

    # Process each rolling mean correlation.
    for i in range(n):
        corr = rolling_mean_corr[i]
        # Find the closest correlation key.
        if corr <= corr_keys[0]:
            corr_index = 0
        elif corr >= corr_keys[-1]:
            corr_index = corr_keys.shape[0] - 1
        else:
            # Loop through the keys to find where corr fits.
            corr_index = 0
            for j in range(1, corr_keys.shape[0]):
                if corr < corr_keys[j]:
                    # Check which key is closer: the one before or this one.
                    if abs(corr - corr_keys[j - 1]) <= abs(corr_keys[j] - corr):
                        corr_index = j - 1
                    else:
                        corr_index = j
                    break
        result[i, 0] = dm_values[corr_index, asset_index]
    return result


spec = [
    ("arr", float64[:, :]),
    ("period", int64),
    ("corr_analyzer", Correlation.class_type.instance_type),
    ("_data", float64[:, :]),
    ("_multiplier", float32[:, :]),
]

@jitclass(spec=spec)
class Multiplier:
    def __init__(self, period: int = 21):
        self.period = period
        self.corr_analyzer = Correlation(period)
    
    def get_multiplier(self, data: np.ndarray) -> np.ndarray:
        return calculate_multiplier(
            self.corr_analyzer.rolling_mean(data),
            data.shape[1]
        )
    
    def get_multiplier_single(self, data: np.ndarray) -> float:
        return calculate_multiplier_single(
            self.corr_analyzer.mean(data),
            data.shape[1]
        )


# ======================================================================================
class DiversificationMultiplier:
    """
    Class to calculate the diversification multiplier.

    'The only free lunch in investing/trading is diversification.' ;)

    The idea is, that with more assets you can take on more risk with
    your single positions because of the diversified risk. The less
    correlated the assets are, the higher the diversification multiplier.

    The multiplier is meant to be applied to:

    1) The leverage/position size that was determined by the position
    sizing algorithm of a strategy which operates on multiple assets.
    The input array should contain the 'close' prices for each asset.

    2) The leverage/position size of multiple strategies for the same
    asset. The input array should contain the the equity/portfolio value
    for the different strategies in this case.

    The matrix in the class definition belwo is used to determine the
    'diversification multiplier'. The key represent the (closest value
    to) the mean correlation of the assets for the lookback period
    (defined by the argument for the init method, default 14). The
    sub-keys stand for the number of assets/strategies. The values are
    the multiplier to apply to the leverage/position size.

    This is taken from Robert Carver´s book: Systematic Trading (p.131)
    """

    DM_MATRIX = {
        0: {2: 1.41, 3: 1.73, 4: 2.0, 5: 2.2, 10: 3.2, 15: 3.9, 20: 4.5, 50: 7.1},
        0.25: {
            2: 1.27,
            3: 1.41,
            4: 1.51,
            5: 1.58,
            10: 1.75,
            15: 1.83,
            20: 1.86,
            50: 1.94,
        },  # noqa: E501
        0.5: {
            2: 1.15,
            3: 1.22,
            4: 1.27,
            5: 1.29,
            10: 1.35,
            15: 1.37,
            20: 1.38,
            50: 1.40,
        },  # noqa: E501
        0.75: {
            2: 1.10,
            3: 1.12,
            4: 1.13,
            5: 1.15,
            10: 1.17,
            15: 1.17,
            20: 1.18,
            50: 1.19,
        },  # noqa: E501
        1.0: {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0, 50: 1.0},
    }

    def __init__(self, data: np.ndarray, period: int = 14):
        self._data = data
        self.period = period
        self.corr_analyzer = Correlation(self.period)

        self._rolling_mean_correlation = self.corr_analyzer.rolling_mean(data)
        self._multiplier = self._calculate_multiplier()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray):
        self._data = new_data
        self._rolling_mean_correlation = self.corr_analyzer.rolling_mean(new_data)
        self._multiplier = self._calculate_multiplier()

    @property
    def multiplier(self) -> np.ndarray:
        return self._multiplier

    def _calculate_multiplier(self) -> np.ndarray:
        if self._data.shape[1] < 2:
            return np.ones(self._data.shape[0], dtype=np.float16)

        choices_for_correlations = np.array(list(self.DM_MATRIX.keys()))
        choices_for_no_of_assets = np.array(list(self.DM_MATRIX[0].keys()))

        no_of_assets = min(self._data.shape[1], 50)
        closest_no_of_assets = choices_for_no_of_assets[
            np.argmin(np.abs(choices_for_no_of_assets - no_of_assets))
        ]

        closest_corr_indices = np.searchsorted(
            choices_for_correlations, self._rolling_mean_correlation, side="left"
        )
        closest_corr_indices = np.clip(
            closest_corr_indices, 0, len(choices_for_correlations) - 1
        )
        closest_corrs = choices_for_correlations[closest_corr_indices]

        result = np.array(
            [self.DM_MATRIX[corr][closest_no_of_assets] for corr in closest_corrs],
            dtype=np.float16,
        ).reshape(-1, 1)
        return result


if __name__ == "__main__":
    m = Multiplier(
        np.random.normal(size=(100, 2)).astype(np.float64)
    )