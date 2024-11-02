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
    s_breakout, s_tema_cross, s_linreg, s_kama_cross, s_trix, trend_1, contra_1
)

symbol = "BTCUSDT"
interval = "1d"

start = -365 * 5  # 'December 01, 2018 00:00:00'
end = 'now UTC'

strategy = s_linreg
risk_levels = [0, 4]
max_leverage = 2
max_drawdown = 50
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


def test_optimize():
    # fetch the OHLCV data from the database
    data = _get_ohlcv_from_db()

    # Optimize the strategy
    best_parameters = optimizer.optimize(
        signal_generator=sig_gen,
        data=data,
        interval=interval,
        risk_levels=risk_levels,
        max_leverage=max_leverage,
        max_drawdown_pct=max_drawdown
    )

    if not best_parameters:
        logger.info('No profitable parameters with acceptable drawdown found.')
        return

    # sort results by sharpe ratio
    best_parameters.sort(
        key=lambda x: x[2]['kalmar_ratio'],
        reverse=True
        )

    profits = [result[2]['profit'] for result in best_parameters]
    logger.info(f'Best profit: {max(profits):.2f}%')
    logger.info(f'Worst profit: {min(profits):.2f}%')

    for result in best_parameters[:25]:
        logger.info(
            "params: %s :: risk level %s :: stats %s",
            result[0],
            result[1],
            {k: round(v, 3) for k, v in result[2].items()}
        )
    logger.info(f'Best parameters length: {len(best_parameters)}')

    return best_parameters


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    # test_optimizer()
    # test_estimate_execution_time()
    # test_estimate_combinations()
    # test_vector_generator()
    test_optimize()
    pass
