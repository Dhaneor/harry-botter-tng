#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import logging
import pandas as pd

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

LOG_LEVEL = "DEBUG"
logger = logging.getLogger('main')
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# ------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# ------------------------------------------------------------------------------------

from src.analysis.strategies import signal_generator as sg  # noqa: E402, F401
from src.analysis.optimizer import Optimizer  # noqa: E402, F401
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    breakout
)

# ---------------------------------------- SETUP -------------------------------------
# Create a signal generator
signal_generator = sg.factory(breakout)
# Create an optimizer
optimizer = Optimizer(signal_generator, 'grid_search')


# ------------------------------------- TESTS ---------------------------------------
def test_optimizer():
    # Load data
    data_path = os.path.join(parent, 'tests', 'data', 'btcusd_1min.csv')
    df = pd.read_csv(data_path)

    # Generate candidate parameters
    candidates = optimizer.generate_candidates()

    # Evaluate candidate parameters
    results = [
        optimizer.optimization_strategy.evaluate_candidate(c) for c in candidates
        ]

    # Update population with results
    optimizer.optimization_strategy.update_population(results)


def test_estimate_combinations():
    # Estimate the number of combinations
    total_combinations = optimizer.estimate_combinations()

    logger.info(f'Total combinations: {total_combinations}')

    return total_combinations


def test_estimate_execution_time():
    # Estimate the execution time
    execution_time = optimizer.estimate_execution_time()

    logger.info(f'Estimated execution time: {execution_time} seconds')

    return execution_time


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    # test_optimizer()
    test_estimate_execution_time()
