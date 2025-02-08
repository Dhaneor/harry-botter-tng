#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 06:09:23 2025

@author: dhaneor
"""

import numpy as np
import pytest
from analysis.models.market_data_store import (
    MarketDataStore, MarketState, MarketStatePool
)
from util import get_logger

logger = get_logger("main", "DEBUG")

# ......................... Tests for the MarketState classes ..........................
def test_market_state_update():
    timestamp = 1735865000000
    open_ = np.random.rand(1000, 10)
    high = np.random.rand(1000, 10)
    low = np.random.rand(1000, 10)
    close = np.random.rand(1000, 10)
    volume = np.random.rand(1000, 10)
    vola_anno = np.random.rand(1000, 10)
    sr_anno = np.random.rand(1000, 10)
    atr = np.random.rand(1000, 10)

    ms = MarketState()

    ms.test_update(
        timestamp,
        open_[0, :],
        high[0, :],
        low[0, :],
        close[0, :],
        volume[0, :],
        vola_anno[0, :],
        sr_anno[0, :],
        atr[0, :],
    )

    assert isinstance(ms, MarketState)
    assert ms.timestamp == timestamp
    
    try:
        np.testing.assert_equal(ms.open, open_[0, :])
        np.testing.assert_equal(ms.high, high[0, :])
        np.testing.assert_equal(ms.low, low[0, :])
        np.testing.assert_equal(ms.close, close[0, :])
        np.testing.assert_equal(ms.volume, volume[0, :])
        np.testing.assert_equal(ms.vola_anno, vola_anno[0, :])
        np.testing.assert_equal(ms.sr_anno, sr_anno[0, :])
        np.testing.assert_equal(ms.atr, atr[0, :])

    except AssertionError as e:
        print(str(e))
        raise


def test_market_state_pool():
    p = MarketStatePool(10)

    assert isinstance(p, MarketStatePool)
    assert len(p.pool) == 10
    assert p.size == 10
    assert isinstance(p.pool[0], MarketState)

def test_pool_get_state():
    p = MarketStatePool(10)
    s = p._get()

    assert isinstance(s, MarketState)
    assert p.size == 9

def test_pool_release_state():
    p = MarketStatePool(10)
    s1 = p._get()
    s2 = p._get()

    p._release(s1)

    assert p.size == 9

    p._release(s2)

    assert p.size == 10

# ........................ Tests for the MarketDataStore class .........................
@pytest.fixture
def sample_data():
    periods = 100
    symbols = 2

    timestamp = np.tile(
        np.asarray([1736500000 + i * 86_400_000 for i in range(periods)]), 
        symbols
    ).reshape(-1, symbols)

    return {
        "timestamp": timestamp,
        "open": np.random.rand(periods, symbols),
        "high": np.random.rand(periods, symbols),
        "low": np.random.rand(periods, symbols),
        "close": np.random.rand(periods, symbols),
        "volume": np.random.rand(periods, symbols),
    } 


def test_market_data_store_initialization(sample_data):
    try:
        mds = MarketDataStore(**sample_data)
    except Exception as e:
        print(e)
        print(f"shape close: {sample_data['close'].shape}")
        raise
    assert isinstance(mds, MarketDataStore)
    assert mds.periods == 100
    assert mds.symbols == 2


def test_periods_per_year(sample_data):
    mds = MarketDataStore(**sample_data)
    assert isinstance(mds.periods_per_year, int)
    assert mds.periods_per_year > 0


def test_compute_atr(sample_data):
    mds = MarketDataStore(**sample_data)
    
    assert mds.atr.shape == sample_data['close'].shape

    try:
        assert not np.isnan(mds.atr[mds.lookback + 1:]).all()
    except AssertionError as e:
        logger.error(e)
        logger.error(mds.atr)
        raise


def test_compute_annualized_volatility(sample_data):
    mds = MarketDataStore(**sample_data)
    assert not np.isnan(mds.vola_anno).all()
    assert mds.vola_anno.shape == sample_data['close'].shape


def test_get_state(sample_data):
    mds = MarketDataStore(**sample_data)
    state = mds._get_state(0)
    assert state.timestamp == sample_data['timestamp'][0][0]
    assert np.array_equal(state.close, sample_data['close'][0])


def test_smooth_it(sample_data):
    mds = MarketDataStore(**sample_data)
    arr = np.random.rand(100, 2)
    arr1 = np.random.rand(100, 2)
    smoothed = mds._smooth_it(arr1, factor=5)
    assert smoothed.shape == arr.shape
    assert not np.array_equal(smoothed, arr)  # Ensure it's actually smoothed


def test_signal_scale_factor(sample_data):
    mds = MarketDataStore(**sample_data)
    assert mds.signal_scale_factor.shape == sample_data['close'].shape
    assert not np.isnan(mds.signal_scale_factor[mds.lookback:]).any()


def test_sr_anno(sample_data):
    mds = MarketDataStore(**sample_data)
    assert mds.sr_anno.shape == sample_data['close'].shape
    assert not np.isnan(mds.sr_anno[mds.lookback:]).any()


def test_custom_lookback(sample_data):
    custom_lookback = 30
    mds = MarketDataStore(**sample_data, lookback=custom_lookback)
    assert mds.lookback == custom_lookback
