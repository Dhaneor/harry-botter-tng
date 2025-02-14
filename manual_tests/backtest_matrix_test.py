#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 14 01:22:23 2025

@author dhaneor
"""
import numpy as np
import time
# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

from analysis.backtest.backtest_nb import BackTestCore, Config
from analysis import MarketData, LeverageCalculator, signal_generator_factory
from util.logger_setup import get_logger
from analysis.strategy.definitions import ema_cross, linreg, rsi  # noqa: F401

logger = get_logger('main', level="DEBUG")


periods = 1_000
assets = 1
strategies = 1


market_data = MarketData.from_random(length=periods, no_of_symbols=assets)

lc = LeverageCalculator(market_data)
leverage = lc.leverage()

signal_generator = signal_generator_factory(rsi)
signal_generator.market_data = market_data

base_signals = signal_generator.execute(compact=True)
# Extend the signals to the specified number of strategies
signals = np.repeat(base_signals, strategies, axis=2)

config = Config(10_000)

if __name__ == "__main__":
    bt = BackTestCore(market_data.mds, leverage, signals, config)
    bt.run()
    signal_generator.randomize()
    signal_generator.execute(compact=True)

    logger.setLevel("ERROR")

    runs = 100

    st = time.time()
    with Profile(timeunit=0.000_001) as p:
        for i in range(runs):
            # signal_generator.randomize()
            # base_signals = signal_generator.execute(compact=True)
            # Extend the signals to the specified number of strategies
            # bt.signals = np.repeat(base_signals, strategies, axis=2)
            bt.run()

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)

    )

    et = time.time() - st
    ips = runs * strategies / et 


    print(f'data: {periods} periods x {assets} assets x {strategies} strategies')
    print(f"periods/s: {periods / et:,.0f}")
    print(f"\navg exc time: {(et * 1_000_000 / runs):,.0f} µs")

    print(f"\n~iter/s (1 core): {ips:>10,.0f}")
    print(f"~iter/s (8 core): {ips * 5:>10,.0f}")
    print(f"~iter/m (8 core): {ips * 5 * 60:>10,.0f}")
