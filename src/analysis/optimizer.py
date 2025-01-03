#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides an optimizer class for strategy parameters.

Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import math
import multiprocessing
import numpy as np
import operator
import pandas as pd
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import reduce, partial
from itertools import product, islice
from typing import TypeVar, Generator, Dict, Any, List, Tuple, Callable, Sequence
from tqdm import tqdm

from . import strategy_backtest as bt
from . import strategy_builder as sb
from .strategy import signal_generator as sg
from .backtest.statistics import calculate_statistics
logger = logging.getLogger('main.optimizer')
logger.setLevel(logging.INFO)

T = TypeVar("T")
OhlcvData = dict[np.ndarray]

multiprocessing.set_start_method('spawn', force=True)

INITIAL_CAPITAL = 10_000  # initial capital for backtesting
RISK_FREE_RATE = 0.04  # risk-free rate for calculating Sharpe Ratio
CHUNK_SIZE = 50  # size of chunks for processing data

PERIODS_PER_YEAR = {
    '1m': 365 * 24 * 60,
    '5m': 365 * 24 * 12,
    '15m': 365 * 24 * 4,
    '30m': 365 * 24 * 2,
    '1h': 365 * 24,
    '2h': 365 * 12,
    '4h': 365 * 6,
    '6h': 365 * 4,
    '8h': 365 * 3,
    '12h': 365 * 2,
    '1d': 365,
    '3d': 365 // 3,
    '1w': 365 // 7,
    '1M': 365 // 12
}


# ================================ Helper Functions I =================================
# Functions to handle parameter vectors
def number_of_combinations(generator: sg.SignalGenerator) -> int:
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
    backtest_fn: Callable, strategy: sb.IStrategy, data: dict[np.ndarray]
) -> float:

    # Run the backtest function once to warm up the JIT compiler
    backtest_fn(
        data=data,
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL
    )

    # Run the backtest 50 times to get an average execution time
    execution_time, runs = 0.0, 100
    for _ in range(runs):
        start_time = time.time()
        _ = backtest_fn(
            data=data,
            strategy=strategy,
            initial_capital=INITIAL_CAPITAL
        )
        execution_time += time.time() - start_time

    return 0.5 * execution_time / runs  # 0.27 estimated speedup parallelization


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


def analyze_parameters(
    parameter_sequence: Sequence[Tuple[Any, ...]]
) -> Tuple[Tuple[Any, ...], ...]:
    if not parameter_sequence:
        return tuple()

    n = len(parameter_sequence[0])  # number of parameters in each tuple

    # Create a list of Counters, one for each parameter position
    counters = [Counter() for _ in range(n)]

    # Count occurrences of each value for each parameter position
    for params in parameter_sequence:
        for i, param in enumerate(params):
            counters[i][param] += 1

    # For each parameter position, get the most common value
    most_common = tuple(counter.most_common(1)[0][0] for counter in counters)

    return most_common


# ============================ Result handling/filtering ==============================
def collect_results(results_list):
    """
    Collects optimizer results, grouping them by parameters and risk_level.

    Parameters:
    results_list
        List of tuples in the format
        ((param1, param2, ...), risk_level, max_leverage, stats_dict)

    Returns:
    collected_results
        Dictionary with keys as (params, risk_level) and values as lists
        of result tuples.
    """
    collected_results = {}

    for result in results_list:
        params, risk_level, max_leverage, stats = result
        key = (params, risk_level)

        # Initialize the list for this key if it doesn't exist
        collected_results.setdefault(key, []).append(result)

    return collected_results


def filter_results(results_list):
    """
    Filters optimizer results to keep only the ones with the lowest max_leverage
    for each unique combination of parameters and risk_level.

    Parameters:
    - results_list: List of tuples in the format
      ((param1, param2, ...), risk_level, max_leverage, stats_dict)

    Returns:
    - filtered_results: List of filtered tuples with the lowest max_leverage
      for each unique (params, risk_level) combination.
    """
    filtered_results = {}

    for result in results_list:
        params, risk_level, max_leverage, stats = result
        key = (params, risk_level)

        # Check if this combination of params and risk_level is already
        # in the dictionary
        if key not in filtered_results:
            filtered_results[key] = result  # Add new entry
        else:
            _, _, existing_max_leverage, existing_stats = filtered_results[key]

            # If the current result has a higher profit but a lower max_leverage,
            # update the existing result with the current result.
            equal_profit = stats["profit"] == existing_stats["profit"]

            if equal_profit:
                if max_leverage < existing_max_leverage:
                    filtered_results[key] = result  # Update with lower max_leverage

    # Extract the filtered results
    return list(filtered_results.values())


def filter_results_by_profit_and_leverage(results_list, rel_tol=1e-9):
    """
    Filters optimizer results to keep.

    For each unique combination of parameters and risk_level,
    and for results with effectively the same profit, only
    the one with the lowest max_leverage.

    Parameters:
    results_list: list
        List of tuples in the format
        ((param1, param2, ...), risk_level, max_leverage, stats_dict)
    rel_tol: float
        Relative tolerance for floating-point comparison of profits.

    Returns:
    filtered_results
        List of filtered tuples.
    """
    # Create a dictionary to collect results for each (params, risk_level)
    results_by_key = defaultdict(list)

    for result in results_list:
        params, risk_level, max_leverage, stats = result
        key = (params, risk_level)
        results_by_key[key].append(result)

    filtered_results = []

    # For each key, process the results
    for key, results in results_by_key.items():
        # List to hold groups of results with the same profit
        profit_groups = []
        profits_seen = []

        for res in results:
            stats = res[3]
            profit = stats['profit']
            # Check if this profit is close to any seen profits
            found = False
            for i, p in enumerate(profits_seen):
                if math.isclose(profit, p, rel_tol=rel_tol):
                    profit_groups[i].append(res)
                    found = True
                    break
            if not found:
                # Start a new group
                profits_seen.append(profit)
                profit_groups.append([res])
        # Now, for each profit group, find the result with lowest max_leverage
        for group in profit_groups:
            # Find the result with the lowest max_leverage
            min_leverage = min(res[2] for res in group)
            # Keep the results with the lowest max_leverage
            min_leverage_results = [res for res in group if res[2] == min_leverage]
            # Add these results to the filtered_results
            filtered_results.extend(min_leverage_results)
    return filtered_results


def results_to_dataframe(results_list):
    """
    Converts a list of optimizer results into a pandas DataFrame.

    Parameters:
    results_list
        List of tuples in the format:
        ((param1, param2, ...), risk_level, max_leverage, stats_dict)

    Returns:
    df
        pandas DataFrame with columns p1, p2, ..., risk_level,
        max_leverage, and stats columns
    """
    # Determine the maximum number of parameters across all results
    max_params = max(len(result[0]) for result in results_list)

    # Collect all possible stats keys to ensure all columns are captured
    all_stats_keys = set()
    for result in results_list:
        stats_dict = result[3]
        all_stats_keys.update(stats_dict.keys())

    rows = []
    for result in results_list:
        params_tuple, risk_level, max_leverage, stats_dict = result
        row = {}
        # Add parameters to the row with column names p1, p2, ...
        for i, param in enumerate(params_tuple):
            row[f'p{i+1}'] = param
        # If there are fewer parameters than max_params, fill the rest with None
        for i in range(len(params_tuple), max_params):
            row[f'p{i+1}'] = None
        # Add risk_level and max_leverage
        row['risk_level'] = risk_level
        row['max_leverage'] = max_leverage
        # Add stats to the row
        for key in all_stats_keys:
            row[key] = stats_dict.get(key, None)
        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)

    # Ensure the columns are ordered:
    # p1, p2, ..., risk_level, max_leverage, stats columns
    param_cols = [f'p{i+1}' for i in range(max_params)]
    other_cols = ['risk_level', 'max_leverage']
    stats_cols = list(all_stats_keys)
    df = df[param_cols + other_cols + stats_cols]

    return df


# ================================ Helper Functions II ================================
# Functions for the parallel execution of backtests
def chunk_parameters(signal_generator, chunk_size=1000):
    param_iterator = vector_generator(signal_generator.parameters)
    while True:
        chunk = tuple(islice(param_iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def chunk_mutations(mutations: tuple, chunk_size=1000):
    mutations_iter = iter(mutations)  # Convert tuple to iterator
    while True:
        chunk = tuple(islice(mutations_iter, chunk_size))
        if not chunk:
            break
        yield chunk


def mutations_for_parameters(params: tuple[int | float, ...]):
    """Mutates the parameters for a given result tuple.

    This function creates a tuple with mutated parameters for a given
    result tuple. The mutated parameters can then be used to evaluate
    the stability of the strategy.

    It takes each of the parameters in the result tuple and:
    - if the parameter is an integer, it creates a tuple with values from
    the range of the parameter minus 10 and the range plus 10 (and removes
    negative values), step size: 1.
    - if the parameter is a float, it creates a tuple with values from
    the range of the parameter minus 0.5 and the range plus 0.5 (and removes
    negative values), step size: 0.1.

    The tuples are collected in a list and the functions returns a generator
    that yields all permutations. For this the vector_generator function is used.

    Parameters:
    -----------
        params: tuple[int | float,...]
            A tuple containing parameters
    """
    # Create a tuple with mutated parameters for each parameter
    mutated_params = []
    for param in params:
        if isinstance(param, (int, np.int64)):
            mutated_params.append(tuple(range(max(param - 5, 1), param + 6)))
        elif isinstance(param, (float, np.float64)):
            mut = tuple(np.arange(max(param - 0.05, 0.01), param + 0.06, 0.01))
            mutated_params.append(tuple(round(p, 3) for p in mut))
        else:
            raise ValueError(f'Unsupported parameter type: {type(param)}')

    # Create all permutations of the mutated parameters
    return tuple(product(*mutated_params))


def worker(
    chunk: tuple[tuple[Any, ...], ...],
    strategy_definition: sb.StrategyDefinition,
    data: dict[np.ndarray],
    risk_level: int,
    max_leverage: float,
    backtest_fn: Callable,
    initial_capital: float,
    periods_per_year: int
) -> List[Tuple[Tuple[Any, ...], int, Dict[str, float]]]:
    """
    Process a chunk of parameter combinations for backtesting and optimization.

    This function creates a signal generator, runs backtests for each parameter
    combination in the given chunk, and returns profitable results that meet
    the specified drawdown criterion.

    Parameters:
    -----------
    chunk : tuple[tuple[Any, ...], ...]
        A chunk of parameter combinations to test.
    condition_definitions : Sequence[object]
        Definitions for creating the signal generator.
    data : dict[np.ndarray]
        Market data for backtesting.
    risk_level : int
        Risk level for the backtesting strategy.
    max_leverage : float
        Maximum allowed leverage.
    max_drawdown_pct : float
        Maximum allowed drawdown percentage.
    backtest_fn : Callable
        Function to perform the backtest.
    initial_capital : float
        Initial capital for backtesting.
    periods_per_year : int
        Number of trading periods per year.

    Returns:
    --------
    List[Tuple[Tuple[Any, ...], int, Dict[str, float]]]
        A list of tuples containing profitable parameter combinations,
        their risk levels, and calculated statistics that meet the
        maximum drawdown criterion.
    """
    strategy = sb.build_strategy(strategy_definition)
    cleanup_threshold = 10 * 1024 * 1024  # 10 MB, adjust as needed
    ohlcv_keys = ['open time', 'open', 'high', 'low', 'close', 'volume']
    results = []
    # seen = set()

    for params in chunk:
        strategy.signal_generator.parameters = params
        data_new = {k: v for k, v in data.items() if k in ohlcv_keys}

        bt_result = backtest_fn(
            strategy=strategy,
            data=data_new,
            risk_level=risk_level,
            initial_capital=initial_capital,
            max_leverage=max_leverage
        )

        equity = bt_result.get('b.value')

        print(data_new.keys())

        # print(f'Portfolio value: {equtiy[-10:]}')

        # last = equity[-1]
        # seen.add(last)

        # print(strategy.signal_generator.parameters, seen)

        results.append(
            (
                params,
                risk_level,
                max_leverage,
                calculate_statistics(equity, 0, periods_per_year)
            )
        )

        if sys.getsizeof(data) > cleanup_threshold:
            data = {k: v for k, v in data.items() if k in ohlcv_keys}

        del bt_result

    return filter_results_by_profit_and_leverage(results)


# ====================================================================================
def optimize(
    strategy: sb.SubStrategy,
    data: OhlcvData,
    interval: str = '1d',
    risk_levels: Iterable[float] = (1,),
    max_leverage_levels: tuple[float, ...] = (1,),
    backtest_fn: Callable = bt.run
) -> List[Tuple[tuple[int | float, ...], int, int, Dict[str, float]]]:
    """Runs backtests for different parameter combinations.

    Parameters:
    -----------
    signal_generator : sg.SignalGenerator
        Signal generator for creating the strategy.
    data : OhlcvData
        A dictionary containing market data for backtesting.
    interval : str, optional
        The trading interval for backtesting. Default is '1d'.
    risk_levels : Iterable[float], optional
        Risk levels for backtesting. Default is (1,).
    max_leverage_levels : tuple[float,...]
        Maximum allowed leverage levels. Default is (1, 1.5, 2, 2.5, 3).
    backtest_fn : Callable, optional
        Function to perform the backtest. Default is `bt.run`.

    Returns:
    --------
    List[Tuple[tuple[int | float,...], int, int, Dict[str, float]]]
        A list of tuples, where each tuple contains a parameter
        combination, the risk level, the maximum leverage level,
        and the backtest statistics. The list is unsorted.
    """
    periods_per_year = PERIODS_PER_YEAR.get(interval, 365)
    combinations = number_of_combinations(strategy.signal_generator) \
        * len(risk_levels) \
        * len(max_leverage_levels)

    est_exc_time = combinations * estimate_exc_time(
        backtest_fn, strategy, data)

    logger.info("Starting parallel optimization...")
    logger.info("leverage levels: %s", max_leverage_levels)
    logger.info("testing %s combinations", combinations)
    logger.info("Estimated execution time: %.2fs", est_exc_time)

    num_cores = 1  # multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)  # Use all cores, but at least 1
    logger.info("Using %s processes", num_processes)

    results = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        start_time = time.time()

        with tqdm(total=combinations, desc="Optimizing") as pbar:
            for max_leverage in max_leverage_levels:
                for risk_level in risk_levels:
                    worker_fn = partial(
                        worker,
                        strategy_definition=strategy.definition,
                        data=data,
                        risk_level=risk_level,
                        backtest_fn=backtest_fn,
                        initial_capital=INITIAL_CAPITAL,
                        max_leverage=max_leverage,
                        periods_per_year=periods_per_year,
                    )

                    chunks_iterator = chunk_parameters(
                        strategy.signal_generator, CHUNK_SIZE
                        )

                    for result in pool.imap_unordered(worker_fn, chunks_iterator):
                        results.extend(result)
                        pbar.update(CHUNK_SIZE)

            pbar.close()
            duration = time.time() - start_time
            logger.info(
                "Optimization completed in %.2fs (%.2f/s)",
                duration,
                combinations / duration
                )

    return results, combinations


# ====================================================================================
def soptimize(
    signal_generator: sg.SignalGenerator,
    data: OhlcvData,
    interval: str = '1d',
    risk_levels: Iterable[float] = (1,),
    max_leverage: float = 1,
    max_drawdown_pct: float = 99,
    backtest_fn: Callable = bt.run
) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:

    periods_per_year = PERIODS_PER_YEAR.get(interval, 365)

    combinations = number_of_combinations(signal_generator) * len(risk_levels)
    est_exc_time = combinations * estimate_exc_time(backtest_fn, signal_generator, data)

    logger.info("Starting optimization...")
    logger.info("testing %s combinations", combinations)
    logger.info("Estimated execution time: %.2fs", est_exc_time)

    results = []
    cleanup_threshold = 100 * 1024 * 1024  # 100 MB, adjust as needed

    # List of keys representing OHLCV data
    ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']

    # Create a tqdm progress bar
    pbar = tqdm(desc="Optimizing", total=combinations, unit=" combinations")

    start_time = time.time()
    combinations_tested = 0

    for risk_level in risk_levels:
        func = partial(
            backtest_fn,
            risk_level=risk_level,
            initial_capital=INITIAL_CAPITAL,
            max_leverage=max_leverage
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
                results.append(
                    (
                        params,
                        risk_level,
                        calculate_statistics(
                            portfolio_values, RISK_FREE_RATE, periods_per_year
                        )
                    )
                )

            # Update progress
            combinations_tested += 1
            if combinations_tested % 100 == 0:  # Update every 100 combinations
                pbar.update(100)

    pbar.close()

    exc_time = time.time() - start_time

    results = [
        pr for pr in results
        if abs(pr[2].get('max_drawdown')) < max_drawdown_pct
        ]

    logger.info("Optimization completed in %.2fs", exc_time)
    logger.info("Found %s profitable results", len(results))
    logger.info("Combinations per second: %.2f", combinations_tested / exc_time)

    return results


def check_robustness(
    signal_generator: sg.SignalGenerator,
    data: OhlcvData,
    params: tuple[int | float, ...],
    interval: str = '1d',
    risk_levels: Iterable[float] = (1,),
    max_leverage_levels: tuple[float, ...] = (1,),
    backtest_fn: Callable = bt.run
):
    """
    Tests the robustness of the strategy by optimizing its parameters and
    backtesting them against a given market data.

    Parameters:
    -----------
    signal_generator : sg.SignalGenerator
    Signal generator for creating the strategy.
    data : OhlcvData
    Market data for backtesting.
    interval : str, optional
    Time interval for backtesting. Default is '1d'.
    risk_levels : Iterable[float], optional
    Risk levels for backtesting. Default is (1,).
    max_leverage : float, optional
    Maximum allowed leverage. Default is 1.

    Returns:
    results : tuple[tuple[int | float], Dict[str, float]]]
    """
    mutations = mutations_for_parameters(params)
    combinations = len(mutations) * len(risk_levels) * len(max_leverage_levels)
    periods_per_year = PERIODS_PER_YEAR.get(interval, 365)

    logger.info("Starting robustness check ...")
    logger.info("testing %s combinations", combinations)

    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)

    results, chunk_size = [], 100

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=combinations, desc="Optimizing") as pbar:
            for max_leverage in max_leverage_levels:
                for risk_level in risk_levels:
                    worker_fn = partial(
                        worker,
                        condition_definitions=signal_generator.condition_definitions,
                        data=data,
                        risk_level=risk_level,
                        backtest_fn=backtest_fn,
                        initial_capital=INITIAL_CAPITAL,
                        max_leverage=max_leverage,
                        periods_per_year=periods_per_year,
                    )

                    chunks = chunk_mutations(mutations, chunk_size)

                    for result in pool.imap_unordered(worker_fn, chunks):
                        results.extend(result)
                        pbar.update(chunk_size)

            pbar.close()

            logger.info(
                "Total profitable results: %s (%.2f)",
                len(results),
                (len(results) / combinations) * 100
            )

    return filter_results(results)


# =====================================================================================
if __name__ == '__main__':
    pass
