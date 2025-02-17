import cProfile
import pstats

import cython
import numpy as np
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.analysis.backtest.backtest_cy import BackTestCore, Config  # noqa: E402
from src.analysis import MarketData, LeverageCalculator, signal_generator_factory  # noqa: E402
from src.analysis.strategy.definitions import ema_cross, linreg, rsi  # noqa: F401, E402

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
        
    def run_multiple_times():
        for _ in range(1_000):  # Adjust this number as needed
            bt.run()
    
    cProfile.runctx('run_multiple_times()', globals(), locals(), 'profile.stats')
    stats = pstats.Stats('profile.stats')
    (
        stats
        .strip_dirs()
        .sort_stats('cumulative')
        .print_stats(30)
    )

