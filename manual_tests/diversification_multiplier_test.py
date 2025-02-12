#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import numpy as np
import time

from analysis.models.market_data import MarketData
from analysis.statistics.correlation import Correlation
from analysis.diversification_multiplier import (
    calculate_multiplier, Multiplier
)

corr = Correlation()

assets = 20
md = MarketData.from_random(180, assets)
data = md.mds.close
data = corr.levels_to_log_returns(data)
runs = 1000


def test_multiplier_single():
    m = Multiplier()

    st = time.time()
    for _ in range(runs):
        ms = m.get_multiplier_single(data)

    avg_exc_time = (time.time() - st) * 1e6 / runs
    
    print(ms)
    print(f"avg exc time: {avg_exc_time:,.0f}µs")


def test_dmc_class():
    dmc = Multiplier()
    dmc.get_multiplier(data)
    
    exc_times = []
    for _ in range(runs):
        st = time.time()
        multiplier = dmc.get_multiplier(data)
        et = (time.time() - st) * 1e6
        exc_times.append(et)

    print("-------------------- DMC class --------------------")
    print(f"Diversification Multiplier (last 5 periods):\n{multiplier[-5:]}")
    
    unique = np.unique(multiplier)
    avg_exc_time = sum(exc_times) / runs

    return multiplier, unique, avg_exc_time


def test_dm_func():
    corr = Correlation(21)

    # initialize Numba functions (compilation)
    correlation = corr.rolling_mean(data)
    multiplier = calculate_multiplier(correlation, assets)
    
    exc_times = []
    for _ in range(runs):
        st = time.time()
        correlation = corr.rolling_mean(data)
        multiplier = calculate_multiplier(correlation, assets)
        et = (time.time() - st) * 1e6
        exc_times.append(et)

    print("--------------- Multiplier function ---------------")
    print(f"Diversification Multiplier (last 5 periods):\n{multiplier[-5:]}")
    
    unique = np.unique(multiplier)
    avg_exc_time = sum(exc_times) / runs

    return multiplier, unique, avg_exc_time

    
def test_compare_implementations():
    dmc = test_dmc_class()
    mfn = test_dm_func()

    same_values = np.array_equal(dmc[0], mfn[0])
    same_unique = np.array_equal(dmc[1], mfn[1])

    print(f"same values: {same_values}")
    print(f"unique values: {dmc[1]} == {mfn[1]} -> {same_unique}")
    print(f"avg exc time: class -> {dmc[2]:,.2f}µs :: func -> {mfn[2]:,.2f}µs")

    try:
        np.testing.assert_array_equal(dmc[0], mfn[0])
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    test_multiplier_single()
    # test_compare_implementations()
