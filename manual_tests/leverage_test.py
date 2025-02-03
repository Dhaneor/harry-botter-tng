#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import time
import numpy as np
import sys  # noqa: F401

# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

from analysis.leverage import LeverageCalculator, Leverage
from analysis.models.market_data import MarketData
from util import seconds_to


length = 1_000
cols = 20  # len(assets)

md = MarketData.from_random(length, cols)

lc = LeverageCalculator(
    market_data=md,
    risk_level=2,
    max_leverage=10,
    smoothing=10
)

lnb = Leverage(market_data=md.mds, risk_level=5)


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
def test_get_leverage():
    leverage = lc.leverage()
    print(leverage[-10:])

def test_get_leverage_jit_class():
    leverage = lnb.leverage()
    print(leverage[-10:])


# -------------------------------------------------------------------------------------
#                                        MAIN                                         #
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    test_get_leverage_jit_class()
    # sys.exit()

    # =================================================================================
    runs = 1000

    start = time.perf_counter()
    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = lnb.leverage()

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    et = time.perf_counter()
    print(f"data: {cols} assets / {len(md)} periods")
    print(f"average execution time: {seconds_to((et - start) / runs)}")
    print(f"iterations per second: {int(runs / ((et - start))):,} iterations/second")
