#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import time
import numpy as np
import sys

# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

from analysis import leverage as lv
from analysis.models.market_data import MarketData, MarketDataStore
from util import execution_time, seconds_to


length = 50
assets = [
    'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'BNBUSDT',
    'XDCUSDT', 'QNTUSDT', 'XLMUSDT'
]
cols = len(assets)

ts = np.arange(length, dtype=np.int64) * 10_000_000,  # 10 seconds intervals

ohlcv = {
    'timestamp': np.random.randint(5,  size=(length, cols)),
    'open_': np.random.rand(length, cols).astype(np.float32),
    'high': np.random.rand(length, cols).astype(np.float32),
    'low': np.random.rand(length, cols).astype(np.float32),
    'close': np.random.rand(length, cols).astype(np.float32),
    'volume': np.random.rand(length, cols).astype(np.float32),
}

periods, assets = ohlcv.get('close').shape
print(f"periods: {periods}, assets: {assets}")

mds = MarketDataStore(**ohlcv)
md = MarketData(mds, ['BTCUSDT'])


# ======================================================================================
def generate_stock_prices(initial_price=100, num_days=365, num_assets=2, volatility=0.01):
    """
    Generate simulated stock prices using a random walk model.

    Parameters:
    -----------
    initial_price : float
        The starting price for the simulation.
    num_days : int
        The number of days to simulate.
    num_assets : int
        The number of assets to simulate.
    volatility : float, optional
        The volatility of the price changes (default is 0.01).

    Returns:
    --------
    np.ndarray
        A 2D array of shape (num_days, num_assets) containing the simulated prices.
    """
    # Generate daily returns
    daily_returns = np.random.normal(0, volatility, (num_days, num_assets))

    # Calculate price path
    price_path = initial_price * np.exp(np.cumsum(daily_returns, axis=0))

    # Ensure the initial price is set correctly
    price_path = np.vstack([np.full(num_assets, initial_price), price_path[1:]])

    return price_path


def get_max_leverage():
    return lv.max_leverage(data=ohlcv, risk_level=3)


@execution_time
def get_diversifcation_multiplier(close_prices: np.ndarray):
    return lv.diversification_multiplier(close_prices=close_prices)


# --------------------------------------------------------------------------------------
@execution_time
def test_vol_anno(close: np.ndarray):
    ohlcv['volatility'] = lv.vol_anno(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
        use_log_returns=False
    )


@execution_time
def test_vol_anno_nb(close: np.ndarray):
    ohlcv['volatility'] = lv.vol_anno_nb(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
    )


@execution_time
def test_conservative_sizing(data: dict):
    lv._conservative_sizing(data=data, target_risk_annual=0.1, smoothing=1)


@execution_time
def test_aggressive_sizing(data: dict):
    lv._aggressive_sizing(data=data, risk_limit_per_trade=0.05)


@execution_time
def test_max_leverage_fast(data: dict):
    ohlcv['leverage'] = lv.max_leverage(data=data, risk_level=1)


def test_get_diversification_multiplier(rows: int = 30, number_of_assets: int = 2):
    dmc = lv.DiversificationMultiplier()
    data = generate_stock_prices(num_days=10_000, num_assets=2, volatility=0.05)

    print(data[-10:])

    for _ in range(5):
        dm = dmc.multiplier(data)

    print(dm[-10:])


# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    test_get_diversification_multiplier(rows=30, number_of_assets=3)
    sys.exit()

    # =================================================================================
    runs = 10_000
    length = 100_000
    data = np.random.rand(length)

    lc = lv.LeverageCalculator(md)

    func = lc.leverage

    # func(3)

    # sys.exit()

    start = time.perf_counter()
    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            func(3)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    # for _ in range(runs):
    #     test_strategy_run(s, False)

    et = time.perf_counter()
    print(f"length data: {length}")
    print(f"average execution time: {seconds_to((et - start) / runs)}")
    print(f"iterations per second: {int(runs / ((et - start))):,} iterations/second")
