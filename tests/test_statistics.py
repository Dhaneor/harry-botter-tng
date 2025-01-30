#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides tests for the Statistics class.

Created on Tue Jan 28 17:33:23 2025

@author dhaneor
"""

import pytest
import numpy as np
from analysis.statistics.statistics import Statistics


@pytest.fixture
def stats():
    return Statistics()

@pytest.fixture
def sample_data():
    return np.array([
        [100, 200, 300],
        [101, 202, 303],
        [103, 205, 308],
        [102, 203, 305],
        [104, 208, 312]
    ], dtype=np.float64)


def test_rolling_mean(stats, sample_data):
    mean = stats.mean(sample_data, 2)
    expected = np.array(
        [
            [np.nan, np.nan, np.nan],
            [100.5, 201.0, 301.5],
            [102.0,  203.5, 305.5],
            [102.5, 204.0,  306.5],
            [103.0,  205.5, 308.5],
        ], 
        dtype=np.float64
    )
    try:
        np.testing.assert_array_almost_equal(mean, expected, decimal=1)
    except Exception as e:
        print(f"Error: {str(e)}")
        print(mean)
        raise pytest.fail("Arrays are not equal")


def test_rolling_std(stats, sample_data):
    std = stats.std(sample_data, 2)
    expected = np.array([
        [np.nan, np.nan, np.nan],
        [0.5, 1.0, 1.5],
        [1.0, 1.5, 2.5],
        [0.5, 1.0, 1.5],
        [1.0, 2.5, 3.5],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(std, expected, decimal=1)


def test_rolling_var(stats, sample_data):
    var = stats.var(sample_data, 2)
    expected = np.array([
        [np.nan, np.nan, np.nan],
        [0.25, 1.0, 2.25],
        [1.0, 2.25, 6.25],
        [0.25, 1.0, 2.25],
        [1.0, 6.25, 12.25],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(var, expected, decimal=1)


def test_rolling_min(stats, sample_data):
    min_val = stats.min(sample_data, 2)
    expected = np.array([
        [np.nan, np.nan, np.nan],
        [100, 200, 300],
        [101, 202, 303],
        [102, 203, 305],
        [102, 203, 305],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(min_val, expected, decimal=1)


def test_rolling_max(stats, sample_data):
    max_val = stats.max(sample_data, 2)
    expected = np.array([
        [np.nan, np.nan, np.nan],
        [101, 202, 303],
        [103, 205, 308],
        [103, 205, 308],
        [104, 208, 312],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(max_val, expected, decimal=1)


def test_rolling_sum(stats, sample_data):
    sum_val = stats.sum(sample_data, 2)
    expected = np.array([
        [np.nan, np.nan, np.nan],
        [201, 402, 603],
        [204, 407, 611],
        [205, 408, 613],
        [206, 411, 617],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(sum_val, expected, decimal=1)


# --------------------------- Tests for non-anualized methods --------------------------
def test_sharpe_ratio(stats, sample_data):
    sr = stats.sharpe_ratio(sample_data)

    returns = sample_data[1:]/ sample_data[:-1] - 1

    print("returns:", returns)
    print("shape returns:", returns.shape)

    mean_returns = np.mean(returns, keepdims=False)
    # mean_returns = np.sum(returns, axis=0) / len(returns)

    print("mean_returns:", mean_returns)
    print("shape mean_returns:", mean_returns.shape)
    
    stdev = np.std(returns, axis=0, keepdims=False)

    print("stdev:", stdev)
    print("shape stdev:", stdev.shape)

    expected = np.asarray([mean_returns / (stdev + 1e-8)])

    try:
        np.testing.assert_array_almost_equal(sr, expected, decimal=3)
    except Exception as e:
        print(e)
        raise


# ----------------------------- Tests for anualized methods ----------------------------
def test_annualized_returns(stats, sample_data):
    ann_returns = stats.annualized_returns(sample_data, 365)
    expected = np.array([[16.515953, 16.515953, 16.515953]], dtype=np.float64)
    np.testing.assert_array_almost_equal(ann_returns, expected, decimal=6)


def test_annualized_volatility(stats, sample_data):
    ann_vol = stats.annualized_volatility(sample_data, 365)
    expected = np.array([[0.228298, 0.237845, 0.232623]], dtype=np.float64)
    np.testing.assert_array_almost_equal(ann_vol, expected, decimal=6)

# def test_annualized_sharpe_ratio(stats, sample_data):
#     sharpe = stats.annualized_sharpe_ratio(sample_data, 365)
#     expected = np.array([[0.405278, 0.210127, 0.138846]], dtype=np.float64)
#     np.testing.assert_array_almost_equal(sharpe, expected, decimal=6)

# def test_atr(stats):
#     high = np.array([[110, 210], [120, 220], [130, 230]], dtype=np.float64)
#     low = np.array([[90, 190], [100, 200], [110, 210]], dtype=np.float64)
#     close = np.array([[100, 200], [110, 210], [120, 220]], dtype=np.float64)
#     period = 2
#     atr = stats.atr(high, low, close, period)
#     expected = np.array([
#         [0, 0],
#         [20, 20],
#         [20, 20]
#     ], dtype=np.float64)
#     np.testing.assert_array_almost_equal(atr, expected, decimal=6)