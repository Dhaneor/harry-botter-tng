#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import time
import logging
import pandas as pd
import numpy as np
from pprint import pprint

# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

from analysis import MarketData, SignalChart
from analysis.strategy import signal_generator as sg
from analysis.strategy.definitions import (  # noqa: F401
    cci, ema_cross, tema_cross, rsi, trix, breakout, kama_cross,
    linreg_roc_btc_1d, linreg_roc_eth_1d, test_er, linreg, aroonosc,
    linreg_ma_cross
)
from helpers_ import get_ohlcv  # noqa: F401
from util import get_logger

logger = get_logger('main', level='DEBUG')

# set interval and length of test data
interval = "2h"
length = 500

sig_def = breakout
data = MarketData.from_random(length=length, no_of_symbols=1)

# ======================================================================================
def _get_test_data(length=1000):
    return {
        'open time': np.arange(length),
        'open': np.random.rand(length),
        'high': np.random.rand(length),
        'low': np.random.rand(length),
        'close': np.random.rand(length),
        'volume': np.random.rand(length)
    }

def test_factory(sig_def, show: bool=True) -> sg.SignalGenerator:
    try:
        sig_gen = sg.signal_generator_factory(sig_def)
    except Exception as exc:
        logger.exception(exc)
        sys.exit()

    assert isinstance(sig_gen, sg.SignalGenerator)

    if show:
        logger.info("created signal generator:\n %s", sig_gen.__dict__)
    return sig_gen


def test_factory_from_existing(sig_gen):
    try:
        sig_gen_new = sg.signal_generator_factory(sig_gen.condition_definitions)
    except Exception as exc:
        logger.exception(exc)
        sys.exit()

    logger.info("created new signal generator from existing: %s", sig_gen_new)
    return sig_gen_new


def test_randomize():
    try:
        sig_gen = sg.signal_generator_factory(sig_def)
    except Exception as exc:
        logger.exception(exc)
        sys.exit()

    logger.info("original signal generator: %s", sig_gen)

    sig_gen.randomize()

    logger.info("randomized signal generator: %s", sig_gen)


def test_get_all_used_indicators():
    sig_gen = test_factory(linreg_roc_btc_1d)

    logger.debug(sig_gen.indicators)

    for n, ind in enumerate(sig_gen.indicators):
        logger.info(ind)
        # logger.debug(
        #     "[%s] %s-> %s -> %s (%s)",
        #     n, ind, ind.parameters, ind.parameter_space, type(ind)
        #     )
        # logger.debug(ind.__dict__)
        # methods = [m for m in dir(ind) if '_' not in m]
        # logger.debug(methods)
        logger.debug("-" * 160)


def test_get_all_parameters():
    sig_gen = test_factory(linreg_roc_btc_1d)

    logger.info(sig_gen.operands)
    logger.info(sig_gen.indicators)

    for param in sig_gen.parameters:
        logger.info(param)
        logger.debug("-" * 160)

def test_get_all_operands():
    """Gets all operands in the SignalGenerator. """
    sig_gen = test_factory(linreg_roc_btc_1d)

    for n, cond in enumerate(sig_gen.conditions):
        for op in filter(lambda x: x is not None, cond.operands):
            logger.debug("[%s] %s", n, op)
            logger.debug("-" * 160)


def test_set_parameters():
    """Tests setting parameters in the SignalGenerator. """
    global data
    sig_gen = sg.signal_generator_factory(sig_def)
    logger.info(sig_gen.parameters)

    params = tuple((p.value for p in sig_gen.parameters))
    logger.info(f"Original parameters: {params}")

    df = pd.DataFrame.from_dict(sig_gen.execute(data))
    print(df)

    keep_keys = [
        'open time', 'human open time', 'open', 'high', 'low', 'close', 'volume'
        ]
    data = {k: data[k] for k in keep_keys}

    for p in sig_gen.parameters:
        p.increase()

    new_params = tuple((p.value for p in sig_gen.parameters))
    logger.info(f"New parameters: {new_params}")

    for c in sig_gen.conditions:
        print(f"Condition: {c}")

    df = pd.DataFrame.from_dict(sig_gen.execute(data))
    print(df)


def test_execute(sig_gen: sg.SignalGenerator = None, show=False, plot=False):
    sig_gen = sig_gen or sg.signal_generator_factory(ema_cross)
    sig_gen.market_data = data  # MarketData.from_random(length=30, no_of_symbols=1)
    signals = sig_gen.execute()

    logger.info("Generated signals: %s", signals)

    if show:
        df = pd.DataFrame.from_dict(data)
        df_plot = df.copy()
        df.replace(0.0, ".", inplace=True)
        df['open time'] = pd.to_datetime(
            df['open time'],
            unit='ms',
            origin='unix',
        )

        df['open time'] = pd\
            .to_datetime(df['open time'])\
            .dt.strftime('%Y-%m-%d %X')

        df.set_index(keys=['open time'], inplace=True)
        print(df.tail(50))

    if plot:
        chart = SignalChart(
            data=df_plot,
            subplots=sig_gen.subplots,
            style='night',
            title=sig_gen.name
            )

        chart.draw()


def test_subplots(sig_gen: sg.SignalGenerator):
    for subplot in sig_gen.subplots:
        logger.info(subplot)


def test_plot(sig_gen: sg.SignalGenerator):
    sig_gen.plot()


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    # test_factory(sig_def)
    # test_randomize()
    # test_execute(None, False, False)

    # test_set_parameters()

    # sig_gen = test_factory_from_existing(sig_gen)

    # sig_gen = test_factory(linreg_roc)
    # test_plot_desc(sig_gen)

    # test_get_all_used_indicators()
    # test_get_all_parameters()
    # test_get_all_operands()

    sig_gen = test_factory(linreg, False)
    sig_gen.market_data = data
    sig_gen.execute()
    # test_subplots(sig_gen)
    test_plot(sig_gen)
    # test_returns(sig_gen, data, True)

    sys.exit()

    logger.setLevel(logging.ERROR)

    runs = 1_000
    sig_gen = test_factory(breakout)
    sig_gen.market_data = data
    sig_gen.execute()

    st = time.time()
    with Profile(timeunit=0.000_001) as p:
        for i in range(runs):
            sig_gen.execute()
            if i % 1 == 0:
                sig_gen.randomize() 

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)

    )

    et = time.time() - st
    ips = runs / et
    periods = len(data["close"]) * ips

    print(f'data: {len(data["close"]):,} periods')
    print(f"periods/s: {periods:,.0f}")
    print(f"\navg exc time: {(et * 1_000_000 / runs):.0f} µs")

    print(f"\n~iter/s (1 core): {ips:>10,.0f}")
    print(f"~iter/s (8 core): {ips * 5:>10,.0f}")
    print(f"~iter/m (8 core): {ips * 5 * 60:>10,.0f}")
