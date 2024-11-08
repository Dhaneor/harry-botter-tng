#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import logging

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
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    s_breakout, s_tema_cross, s_linreg, s_kama_cross, s_trix,
    trend_1, contra_1, s_test_er
)

symbol = "BTCUSDT"
interval = "1d"

start = int(-365*6)
end = 'now UTC'

strategy = s_breakout
risk_levels = [2, 3, 4, 5, 6]
max_leverage_levels = 1,  # (1, 1.25, 1.5, 1.75, 2)
max_drawdown = 25
initial_capital = 10_000 if symbol.endswith('USDT') else 0.5


strategy: sb.CompositeStrategy = sb.build_strategy(strategy)

# ---------------------------------------- SETUP -------------------------------------
# Create a signal generator
sub_strategy: sb.SubStrategy = [v for v in strategy.sub_strategies.values()][0][0]
# print(sub_strategy)

sig_gen = sub_strategy._signal_generator


def _get_ohlcv_from_db():
    hermes = Hermes(exchange='kucoin', mode='backtest')

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
    )

    if res.get('success'):
        df = res.get('message')
        return {col: df[col].to_numpy() for col in df.columns}

    else:
        error = res.get('error', 'no error provided in response')
        raise Exception(error)

    del hermes


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


def test_optimize():
    # fetch the OHLCV data from the database
    data = _get_ohlcv_from_db()

    # Optimize the strategy
    best_parameters = optimizer.optimize(
        signal_generator=sig_gen,
        data=data,
        interval=interval,
        risk_levels=risk_levels,
        max_leverage_levels=max_leverage_levels
    )

    if not best_parameters:
        logger.info('No profitable parameters with acceptable drawdown found.')
        return

    best_parameters = [
        res for res in best_parameters if res[3]['max_drawdown'] > max_drawdown * -1
        ]

    # sort results by kalmar ratio
    best_parameters.sort(
        key=lambda x: x[3]['kalmar_ratio'],
        reverse=True
        )

    for result in best_parameters[:50]:
        logger.info(
            "params: %s :: risk level %s :: max leverage %s, stats %s",
            tuple(round(elem, 4) for elem in result[0]),
            result[1],
            result[2],
            {k: round(v, 3) for k, v in result[3].items()}
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

    df = optimizer.results_to_dataframe(best_parameters)
    print(df.describe())

    return best_parameters


def test_check_robustness():
    best_result = test_optimize()[0]

    results = optimizer.check_robustness(
        signal_generator=sig_gen,
        data=_get_ohlcv_from_db(),
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


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    # test_optimizer()
    # test_estimate_execution_time()
    # test_estimate_combinations()
    # test_vector_generator()
    # test_mutations_for_parameters()
    # test_optimize()
    test_check_robustness()
