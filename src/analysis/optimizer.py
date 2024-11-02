#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides an optimizer class for strategy parameters.

Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import numpy as np
import operator
import sys
import time
from collections.abc import Iterable
from functools import reduce, partial
from itertools import product
from typing import TypeVar, Generator, Dict, Any, List, Tuple, Callable
from tqdm import tqdm

from . import strategy_backtest as bt
from .strategies.signal_generator import SignalGenerator
from .backtest.statistics import calculate_statistics
logger = logging.getLogger('main.optimizer')
logger.setLevel(logging.INFO)

T = TypeVar("T")
OhlcvData = dict[np.ndarray]

TIME_FOR_ONE_BACKTEST = 6  # execution time for one backtest in milliseconds
INITIAL_CAPITAL = 10_000  # initial capital for backtesting


# ================================ Helper Functions ===================================
def number_of_combinations(generator: SignalGenerator) -> int:
    total_combinations = 1

    for indicator in generator.indicators:
        # Get the number of possible values for each parameter
        param_counts = [len(list(param)) for param in indicator.parameters]

        # Multiply the counts to get the number of combinations for this indicator
        indicator_combinations = reduce(operator.mul, param_counts, 1)

        # Multiply with the total
        total_combinations *= indicator_combinations

    return total_combinations


def estimate_exc_time(
    backtest_fn: Callable, signal_generator: SignalGenerator, data: dict[np.ndarray]
) -> float:

    # Run the backtest function once to warm up the JIT compiler
    backtest_fn(
        data=data,
        strategy=signal_generator,
        initial_capital=INITIAL_CAPITAL
    )

    # Run the backtest 50 times to get an average execution time
    execution_time, runs = 0.0, 50
    for _ in range(runs):  # run 100 times to get an average execution time
        start_time = time.time()
        _ = backtest_fn(
            data=data,
            strategy=signal_generator,
            initial_capital=INITIAL_CAPITAL
        )
        execution_time += time.time() - start_time

    return execution_time / runs


def vector_generator(
    iterables: list[Iterable[float | int]]
) -> Generator[list[T], None, None]:
    """
    Generator that yields all possible combinations of values from the given iterables.

    Parameters:
    ----------
    iterables : list[Iterable[float | int]]
        List of iterables, each containing a sequence of values.

    Yields:
    -------
    list[T]
        A combination of values, one from each of the input iterables.
    """
    for combination in product(*iterables):
        yield tuple(combination)


def vector_diff(vector1: list[T], vector2: list[T]) -> list[T]:
    """
    Difference between two vectors.

    Parameters:
    ----------
    vector1 : list[T]
    First vector.
    vector2 : list[T]
    Second vector.
    Returns:
    list[T]
    Difference between vector1 and vector2.
    """
    return [v1 - v2 for v1, v2 in zip(vector1, vector2)]


def optimize(
    signal_generator: SignalGenerator,
    data: OhlcvData,
    risk_levels: Iterable[float] = (1,),
    max_drawdown_pct: float = 99,
    backtest_fn: Callable = bt.run
) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:

    combinations = number_of_combinations(signal_generator) * len(risk_levels)
    est_exc_time = combinations * estimate_exc_time(backtest_fn, signal_generator, data)

    logger.info("Starting optimization...")
    logger.info("testing %s combinations", combinations)
    logger.info("Estimated execution time: %.2fs", est_exc_time)

    profitable_results = []
    cleanup_threshold = 100 * 1024 * 1024  # 100 MB, adjust as needed

    # List of keys representing OHLCV data
    ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']

    # Create a tqdm progress bar
    pbar = tqdm(desc="Optimizing", total=combinations, unit=" combinations")

    start_time = time.time()
    combinations_tested = 0

    for risk_level in risk_levels:
        func = partial(
            backtest_fn, risk_level=risk_level, initial_capital=INITIAL_CAPITAL
        )

        for params in vector_generator(signal_generator.parameters):
            # Check if cleanup is needed
            if sys.getsizeof(data) > cleanup_threshold:
                data = {k: v for k, v in data.items() if k in ohlcv_keys}

            # set the parameters for the signal generator
            for param, value in zip(signal_generator.parameters, params):
                param.value = value

            # Run backtest
            portfolio_values = func(strategy=signal_generator, data=data).get('b.value')

            # If profitable, store the result
            if portfolio_values[-1] > INITIAL_CAPITAL:
                profitable_results.append(
                    (params, risk_level, calculate_statistics(portfolio_values))
                )

            # Update progress
            combinations_tested += 1
            if combinations_tested % 100 == 0:  # Update every 100 combinations
                pbar.update(100)

    pbar.close()

    exc_time = time.time() - start_time

    profitable_results = [
        pr for pr in profitable_results
        if abs(pr[2].get('max_drawdown')) < max_drawdown_pct
        ]

    logger.info("Optimization completed in %.2fs", exc_time)
    logger.info("Found %s profitable results", len(profitable_results))
    logger.info("Combinations per second: %.2f", combinations_tested / exc_time)

    return profitable_results


# =====================================================================================
if __name__ == '__main__':
    pass
