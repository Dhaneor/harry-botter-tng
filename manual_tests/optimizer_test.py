#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import time
import logging
import numpy as np
import warnings  # noqa: F401

# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

from staff.hermes import Hermes
from analysis import strategy_builder as sb
from analysis import optimizer
from analysis.backtest import statistics
from analysis.indicators import indicators_custom
from analysis.strategy.definitions import (  # noqa: F401
    s_breakout, s_tema_cross, s_linreg, s_kama_cross, s_trix,
    s_trend_1, contra_1, s_test_er, s_linreg_ma_cross, s_aroon_osc,
    s_test_ema_cross
)
from util import get_logger, seconds_to

logger = get_logger('main', level="INFO")

# numpy warnings to exceptions
warnings.filterwarnings('error')
# warnings.filterwarnings('always')

symbol = "BTCUSDT"
interval = "1d"

start = "6 years ago UTC"
end = "now UTC"

strategy = s_trend_1
risk_levels = 5,  # [0, 4, 5, 6, 7, 8, 9]
max_leverage_levels = 1,  # (0.75, 1, 1.25, 1.5, 1.75, 2, 2.5)
max_drawdown = 50
initial_capital = 10_000 if symbol.endswith('USDT') else 0.1


strategy: sb.CompositeStrategy = sb.build_strategy(strategy)

# ---------------------------------------- SETUP -------------------------------------
# Create a signal generator
sub_strategy: sb.SubStrategy = [v for v in strategy.sub_strategies.values()][0][0]
sig_gen = sub_strategy.signal_generator
# print(sub_strategy)

noise = indicators_custom.EfficiencyRatio()


def _get_ohlcv_from_db():
    hermes = Hermes(exchange='binance', mode='backtest')

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
    )

    if res.get('success'):
        df = res.get('message')
        df = df.ffill()  # df.fillna(method='ffill')

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

    sub_strategy = next((v for v in list(strategy.sub_strategies.values())[0]))

    # Optimize the strategy
    results, combinations = optimizer.optimize(
        strategy=sub_strategy,
        data=data,
        interval=interval,
        risk_levels=risk_levels,
        max_leverage_levels=max_leverage_levels
    )

    # filter out results with:
    # a) drawdown greater than max_drawdown,
    # b) max drawdown equal to 0, which means that no trades occured
    best_parameters = [
        res for res in results if 0 > res[3]['max_drawdown'] > max_drawdown * -1
    ]

    if not best_parameters:
        logger.info('No profitable parameters with acceptable drawdown found.')
        return

    # sort results by kalmar ratio
    best_parameters.sort(key=lambda x: x[3]['profit'], reverse=True)

    for result in best_parameters[:50]:
        logger.info(
            "params: %s :: risk level %s :: max leverage %s, stats %s",
            tuple(round(elem, 4) for elem in result[0]),
            result[1],
            result[2],
            {k: round(v, 3) for k, v in result[3].items()}
        )

    profitable = sum([1 for result in results if result[3]['profit'] > 0])

    logger.info(
        "Total profitable results: %s (%.2f)",
        profitable, profitable / combinations * 100
    )

    logger.info(
        'Best parameters with less than %s percent drawdown: %s',
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

    # display the parameters for the best profit, which otherwise might not be displayed
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


# =================================================================================== #
#                                   MAIN                                              #
# =================================================================================== #
if __name__ == '__main__':
    # test_optimizer()
    # test_estimate_execution_time()
    # test_estimate_combinations()
    # test_vector_generator()
    # test_mutations_for_parameters()
    test_optimize()
    # test_check_robustness()

    # profile_function()

    sys.exit(0)

    # ================================ Profiling =====================================
    data = np.multiply(np.random.rand(1_000), 10)
    _ = statistics.calculate_statistics(
        portfolio_values=data,
        periods_per_year=optimizer.PERIODS_PER_YEAR[interval]
    )

    logger.info("length data: {len(data)} periods")
    logger.info(data)

    logger.setLevel(logging.ERROR)

    runs = 10_000

    st = time.perf_counter()

    with Profile() as p:
        for _ in range(runs):
            # Optimize the strategy
            _ = statistics.calculate_statistics(
                portfolio_values=data,
                periods_per_year=optimizer.PERIODS_PER_YEAR[interval]
            )

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(50)
    )

    exc_time_str = seconds_to(((time.perf_counter() - st)/runs))
    print(f'length data: {len(data)} periods')
    print(f"average execution time: {exc_time_str}")
