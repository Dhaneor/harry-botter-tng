#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import time
import logging
import numpy as np
import warnings


# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()

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

from src.staff.hermes import Hermes  # noqa: E402, F401
from src.analysis import strategy_builder as sb  # noqa: E402, F401
from src.analysis.strategies import signal_generator as sg  # noqa: E402, F401
from src.analysis import optimizer  # noqa: E402, F401
from src.analysis.indicators import indicators_custom  # noqa: E402, F401
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    s_breakout, s_tema_cross, s_linreg, s_kama_cross, s_trix,
    trend_1, contra_1, s_test_er, s_linreg_ma_cross, s_aroon_osc,
    s_test_ema_cross
)

# numpy warnings to exceptions
# warnings.filterwarnings('error')

symbol = "BTCUSDT"
interval = "1d"

start = "3 years ago UTC"  # int(-365*6)
end = "now UTC"  # 'December 20, 2024 00:00:00'

strategy = s_linreg_ma_cross
risk_levels = [0, 4, 5, 6, 7, 8, 9]
max_leverage_levels = (0.75, 1, 1.25, 1.5, 1.75, 2, 2.5)
max_drawdown = 30
initial_capital = 10_000 if symbol.endswith('USDT') else 0.1


strategy: sb.CompositeStrategy = sb.build_strategy(strategy)

# ---------------------------------------- SETUP -------------------------------------
# Create a signal generator
sub_strategy: sb.SubStrategy = [v for v in strategy.sub_strategies.values()][0][0]
# print(sub_strategy)

sig_gen = sub_strategy._signal_generator
noise = indicators_custom.EfficiencyRatio()


def _get_ohlcv_from_db():
    hermes = Hermes(exchange='binance', mode='backtest')

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
    )

    if res.get('success'):
        df = res.get('message')
        return {col: df[col].to_numpy() for col in df.columns if col != 'close time'}

    else:
        error = res.get('error', 'no error provided in response')
        raise Exception(error)


# ------------------------------------- TESTS ---------------------------------------
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


def test_vector_generator():
    # Generate a vector of combinations
    parameters = sig_gen.parameters
    vector = list(optimizer.vector_generator(parameters))

    logger.info(f'Generated vector: {vector}')
    logger.info(f'Vector length: {len(vector)}')

    return vector


def test_mutations_for_parameters():
    params = (p.value for p in sig_gen.parameters)

    # Generate mutations for the parameters
    mutations = optimizer.mutations_for_parameters(params)

    logger.info(f'Generated mutations: {mutations}')
    logger.info(f'Mutations length: {len(mutations)}')

    return mutations


def test_optimize(data: dict | None = None):
    # fetch the OHLCV data from the database
    data = data or _get_ohlcv_from_db()
    for key, array in data.items():
        try:
            if np.isnan(array).any() or np.isinf(array).any():
                print(f"Warning: {key} contains NaN or inf values")
                sys.exit()
        except TypeError:
            print(f"Warning: {key} is not a numpy array ({type(key)})")
            sys.exit()

    # Optimize the strategy
    results = optimizer.optimize(
        signal_generator=sig_gen,
        data=data,
        interval=interval,
        risk_levels=risk_levels,
        max_leverage_levels=max_leverage_levels
    )

    profitable = []
    for result in results:
        if result[3][-1] > initial_capital:
            portfolio_values = result[3]
            profitable.append([
                result[0],
                result[1],
                result[2],
                optimizer.calculate_statistics(
                    portfolio_values=portfolio_values,
                    periods_per_year=optimizer.PERIODS_PER_YEAR[interval]
                ),
            ])

    if not profitable:
        logger.info('No profitable parameters with acceptable drawdown found.')
        return

    profitable = optimizer.filter_results_by_profit_and_leverage(profitable)

    best_parameters = [
        res for res in profitable if res[3]['max_drawdown'] > max_drawdown * -1
        ]

    # sort results by kalmar ratio
    best_parameters.sort(
        key=lambda x: x[3]['kalmar_ratio'],
        reverse=True
        )

    # filter out results with max drawdown equal to 0, which means that no trades
    # occured during the given timeframe
    best_parameters = [res for res in best_parameters if res[3]['max_drawdown'] != 0.0]

    for result in best_parameters[:50]:
        logger.info(
            "params: %s :: risk level %s :: max leverage %s, stats %s",
            tuple(round(elem, 4) for elem in result[0]),
            result[1],
            result[2],
            {k: round(v, 3) for k, v in result[3].items()}
        )

    logger.info(
        "Total profitable results: %s (%.2f)",
        len(profitable),
        (len(profitable) / len(results)) * 100
    )

    logger.info(
        'Best parameters with less than %s percent drawdown length: %s',
        max_drawdown, {len(best_parameters)}
        )

    # Extract just the parameter tuples from the results
    param_tuples = [result[0] for result in best_parameters[:50]]

    # Analyze the parameters
    most_common_params = optimizer.analyze_parameters(param_tuples)
    logger.info(f"Most common parameter values in top 50: {most_common_params}")

    profits = [result[3]['profit'] for result in best_parameters]
    logger.info(f'Best profit: {max(profits):.2f}%')
    logger.info(f'Worst profit: {min(profits):.2f}%')
    logger.info(f'Average profit: {sum(profits) / len(profits):.2f}%')

    # display the parameters for the best profit, which otherwise might not be disaplyed
    # because parameters with less profit had a better Kalmar ratio
    # for result in best_parameters:
    #     if result[3]['profit'] == max(profits):
    #         logger.info(f'Best profit parameters: {result}')

    df = optimizer.results_to_dataframe(best_parameters)
    print(df.describe())

    return best_parameters


def test_check_robustness():
    data = _get_ohlcv_from_db()

    if test_optimize(data):
        best_result = test_optimize(data)[0]
    else:
        logger.info("No profitable parameters found.")
        return

    results = optimizer.check_robustness(
        signal_generator=sig_gen,
        data=data,
        params=best_result[0],
        interval=interval,
        risk_levels=(best_result[1],),
        max_leverage_levels=(best_result[2],),
    )

    results.sort(
        key=lambda x: x[3]['profit'],
        reverse=True
        )

    for result in results[:50]:
        logger.info(
            "params: %s :: risk level %s :: max leverage %s, stats %s",
            tuple(round(elem, 4) for elem in result[0]),
            result[1],
            result[2],
            {k: round(v, 3) for k, v in result[3].items()}
        )

    profits = [result[3]['profit'] for result in results]
    logger.info(f'Best profit: {max(profits):.2f}%')
    logger.info(f'Worst profit: {min(profits):.2f}%')

    logger.info("============================================================")
    logger.info(best_result)

    df = optimizer.results_to_dataframe(results)
    print(df.describe())

    noise.parameters[0].hard_max = (len(data['close'] + 5))
    noise._parameters[0].value = (len(data['close'] - 5))
    er = noise._noise_index_numpy(data['close'])
    logger.info("Efficiency Ratio: %s", er[-1])


def profile_function(runs=1):
    logger.setLevel(logging.ERROR)
    data = _get_ohlcv_from_db()
    st = time.perf_counter()

    with Profile() as p:
        for _ in range(runs):
            # Optimize the strategy
            _ = optimizer.soptimize(
                signal_generator=sig_gen,
                data=data,
                interval=interval,
                risk_levels=risk_levels,
            )

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(50)
    )

    et = time.perf_counter()
    print(f'length data: {len(data["close"])} periods')
    print(f"average execution time: {((et - st)*1_000/runs):.2f} milliseconds")


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    # test_optimizer()
    # test_estimate_execution_time()
    # test_estimate_combinations()
    # test_vector_generator()
    # test_mutations_for_parameters()
    test_optimize()
    # test_check_robustness()

    # profile_function()
