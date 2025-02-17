#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 14 01:22:23 2025

@author dhaneor
"""
import numpy as np
import time
import os
import sys
# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.analysis.backtest.backtest import BackTestCore, Config  # noqa: E402
from src.analysis import MarketData, LeverageCalculator, signal_generator_factory  # noqa: E402
from src.util.logger_setup import get_logger  # noqa: E402
from src.analysis.strategy.definitions import ema_cross, linreg, rsi  # noqa: F401, E402

logger = get_logger('main', level="ERROR")


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

    runs = 1_000

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
