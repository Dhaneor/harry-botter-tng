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
from analysis.statistics import correlation as corr
from analysis.models.market_data import MarketData, MarketDataStore
from util import seconds_to, execution_time


length = 50
assets = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "XDCUSDT",
    "QNTUSDT",
    "XLMUSDT",
]
cols = len(assets)

ts = (np.arange(length, dtype=np.int64) * 10_000_000,)  # 10 seconds intervals

ohlcv = {
    "timestamp": np.random.randint(5, size=(length, cols)),
    "open_": np.random.rand(length, cols).astype(np.float32),
    "high": np.random.rand(length, cols).astype(np.float32),
    "low": np.random.rand(length, cols).astype(np.float32),
    "close": np.random.rand(length, cols).astype(np.float32),
    "volume": np.random.rand(length, cols).astype(np.float32),
}

mds = MarketDataStore(**ohlcv)
md = MarketData(mds, ["BTCUSDT"])


# ======================================================================================
def generate_stock_prices(
    initial_price=100, num_days=365, num_assets=2, volatility=0.01
):
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
        A 2D array of shape (num_days, num_assets) containing the 
        simulated prices.
    """
    # Generate daily returns
    daily_returns = np.random.normal(0, volatility, (num_days, num_assets))

    # Calculate price path
    price_path = initial_price * np.exp(np.cumsum(daily_returns, axis=0))

    # Ensure the initial price is set correctly
    price_path = np.vstack([np.full(num_assets, initial_price), price_path[1:]])

    return price_path


# --------------------------------------------------------------------------------------

@execution_time
def test_get_diversification_multiplier(data):
    dmc = lv.DiversificationMultiplier()
    multiplier = dmc.multiplier(data)

    print(f"simulated prices for {number_of_assets} assets:\n {data[-2:]}")
    print('-' * 120)
    print(f"Diversification Multiplier (last 10 days): {multiplier[-11:]}")
    print(f"unique values: {np.unique(multiplier)}")
    print(f"min: {np.min(multiplier)}, max: {np.max(multiplier)}")


def test_rolling_correlation():
    @execution_time
    def numpy_version(arr):
        return corr.rolling_mean_correlation(arr, period=20)

    @execution_time
    def numba_version(arr, period=20):
        return corr.rolling_mean_correlation_nb(arr, period=20)

    arr = np.random.rand(100, 3)

    for _ in range(5):
        np_ = numpy_version(arr)
        assert np_ is not None

    print('-' * 80)

    for _ in range(5):
        nb_ = numba_version(arr)
        assert nb_ is not None

    print(f"Numpy version: {np_[-11:]}")
    print(f"Numba version: {nb_[-11:]}")

    assert np.allclose(numpy_version(arr), numba_version(arr)), \
        f"Results are not equal:\n {np.round(np.diff(np.subtract(np_, nb_)), 3)}" \


# -------------------------------------------------------------------------------------
#                                        MAIN                                         #
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    number_of_assets = 20
    number_of_days = 20_000

    data = generate_stock_prices(
        num_days=number_of_days, num_assets=number_of_assets, volatility=0.05
        )
    
    # test_get_diversification_multiplier(data=data)
    # test_rolling_correlation()
    # sys.exit()

    # =================================================================================
    runs = 1000

    lc = lv.LeverageCalculator(md)
    
    # sys.exit()
    dmc = lv.DiversificationMultiplier(data=data)
    print(dmc.multiplier[-11:])
    # sys.exit()

    start = time.perf_counter()
    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = dmc.multiplier

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    et = time.perf_counter()
    print(f"data: {number_of_assets} assets / {len(data)} periods")
    print(f"average execution time: {seconds_to((et - start) / runs)}")
    print(f"iterations per second: {int(runs / ((et - start))):,} iterations/second")
