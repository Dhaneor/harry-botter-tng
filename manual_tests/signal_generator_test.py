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
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint  # noqa: E402, F401
from typing import Hashable
from scipy.stats import norm

# profiler imports
from cProfile import Profile
from pstats import SortKey, Stats

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.analysis.strategies import signal_generator as sg  # noqa: E402
from src.analysis.strategies import condition as cn  # noqa: E402
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    cci, ema_cross, tema_cross, rsi, trix, breakout, kama_cross,
    linreg_roc_btc_1d, linreg_roc_eth_1d, test_er, linreg, aroonosc,
    linreg_ma_cross
)
from src.analysis.chart.tikr_charts import SignalChart  # noqa: E402
from helpers_ import get_ohlcv  # noqa: E402, F401

logger = logging.getLogger("main")
logger.setLevel(logging.ERROR)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# set interval and length of test data
interval = "1d"
length = 365*6

sig_def = linreg_ma_cross

# ======================================================================================
try:
    df = get_ohlcv(symbol="BTCUSDT", interval=interval, as_dataframe=True)
except Exception as e:
    logger.error(f"Error fetching data: {e}")
    sys.exit()

data = {col: df[col].to_numpy() for col in df.columns}

# ======================================================================================


def test_signal_definition(show=False):
    timeperiod = 14

    try:
        open_long = cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "cci",
                {"timeperiod": timeperiod},
            ),
            operand_b={'oversold': -150},
            open_long=cn.COMPARISON.CROSSED_ABOVE,
        )

        open_short = cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "cci",
                {"timeperiod": timeperiod},
            ),
            operand_b={'overbought': 150},
            open_short=cn.COMPARISON.CROSSED_BELOW,
        )
        success = True
    except TypeError as e:
        logger.error(e)
        success = False

    logger.info("open long: %s", open_long)
    logger.info("open short: %s", open_short)
    logger.info("operands are equal: %s", (open_long.operand_a == open_short.operand_a))
    logger.info("operand is hashable: %s", isinstance(open_long.operand_a, Hashable))

    if success:
        sig_def = sg.SignalsDefinition(
            name='test name',
            conditions=[
                open_long,
                open_short,
            ],
        )

        if show:
            print(sig_def)
        return sig_def
    else:
        logger.error("unable to create signal definition")


def test_factory(sig_def):
    try:
        sig_gen = sg.factory(sig_def)
    except Exception as exc:
        logger.exception(exc)
        sys.exit()

    logger.info("created signal generator: %s", sig_gen)
    logger.info("subplots: %s", sig_gen.subplots)
    return sig_gen


def test_factory_from_existing(sig_gen):
    try:
        sig_gen_new = sg.factory(sig_gen.condition_definitions)
    except Exception as exc:
        logger.exception(exc)
        sys.exit()

    logger.info("created new signal generator from existing: %s", sig_gen_new)
    return sig_gen_new


def test_execute(sig_gen: sg.SignalGenerator, data, weight, show=False, plot=False):
    data = sig_gen.execute(data)
    logger.info(sig_gen)

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


def test_plot(sig_gen: sg.SignalGenerator, data):
    sig_gen.plot(data)


# -----------------------------------------------------------------------------
def test_returns(sig_gen: sg.SignalGenerator, data, show=False):

    returns, returns_final = [], []

    for _ in range(1):

        global df

        length = 150_000
        start = random.randint(0, len(df) - length)
        end = start + length

        data = {col: df[start:end][col].to_numpy() for col in df.columns}

        data = sig_gen.execute(data)

        if show:
            df1 = pd.DataFrame.from_dict(data)

            try:
                df1['signal_cont'] = (
                    np.where(
                        df1['s.open_long'], 1, np.where(
                            df1['s.open_short'], -1, np.where(
                                df1['s.close_long'] | df1['s.close_short'], 0, np.nan
                            )
                        )
                    )
                )
            except Exception as exc:
                logger.exception(exc)
                print(df1)
                sys.exit()

            df1['signal_cont'] = df1['signal_cont'].ffill()

            df1['change'] = np.where(
                df1.signal_cont == df1.signal_cont.shift(), False, True
            )
            df1.loc[(df1.change), 'signal'] = df1.signal_cont

            df1['position'] = np.where(df1.signal_cont.shift() == 1, 1, -1)

            df1.loc[(df1.signal.shift() == 1), 'buy'] = df1.open
            df1.loc[(df1.signal.shift() == -1), 'sell'] = df1.open

            # Calculate log returns
            df1['log_return'] = np.log(df1['close'] / df1['close'].shift(1))

            # .. and the normal returns
            df1['returns'] = (df1.close.pct_change() * 100).round(2)

            # Create a unique trade ID for each trade/position
            df1['trade_id'] = \
                (np.random.rand(df1['close'].shape[0]) * 10_000)\
                .astype(int)

            # Adjust the trade IDs based on position changes
            df1.loc[df1['position'] == df1['position'].shift(), 'trade_id'] = np.nan
            df1['trade_id'].ffill(inplace=True)

            # Calculate cumulative returns for each trade/position
            df1['pos_returns'] = np.exp(
                df1.groupby('trade_id')['log_return'].cumsum()
            ) - 1
            df1.pos_returns = ((df1.pos_returns * df1.position) * 100).round(2)

            # remove all returns except the final one for each trade/position
            df1['pos_returns_final'] = df1.pos_returns
            df1.loc[
                ~(df1['position'] != df1['position'].shift(-1)),
                'pos_returns_final'
            ] = np.nan

            returns.extend(df1["pos_returns"].dropna().tolist())
            returns_final.extend(df1["pos_returns_final"].dropna().tolist())

    # # make it nicer
    # drop_cols = [
    #     'open time', 'signal_cont', 'change',
    #     'trade_id', 'log_return', 'pos_returns'
    # ]
    # df1.drop(drop_cols, axis=1, inplace=True)
    # df1.replace(False, '.', inplace=True)
    # df1.replace(np.nan, '.', inplace=True)
    # df1.position.replace(1, 'LONG', inplace=True)
    # df1.position.replace(-1, 'SHORT', inplace=True)

    # pd.set_option('display.max_rows', 500)

    # print('\n' + '~_*_~' * 25)
    # print(df1[-100:])

    if returns:

        plt_ = pd.DataFrame.from_dict(
            {'returns_final': returns_final}
        ).round(1).astype(float)

        plt_.plot(
            kind='hist', bins=100, density=True, figsize=[24, 12],
            alpha=0.5, title='Returns distribution', facecolor='black'
        )

        # Fit a normal distribution to the data:
        # mean and standard deviation
        mu, std = norm.fit(plt_.returns_final)

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=0.5)

        print(plt_.returns_final.describe(percentiles=[p / 10 for p in range(1, 10)]))

        print(
            df1.pos_returns
            .dropna()
            .pct_change()
            .mul(100)
            .round(2)
            .describe(percentiles=[p / 10 for p in range(1, 10)])
        )

        ret = pd.DataFrame.from_dict({'pos_returns': returns}).round(2)

        plt.hist(
            ret.pos_returns.dropna(),
            bins=100,
            density=True,
            alpha=0.5,
            label='pos_returns',
        )

        plt.show()
    else:
        logger.error("unable to calculate returns")
        print(df1.tail(25))


def test_get_all_used_indicators():
    sig_gen = test_factory(linreg_roc_btc_1d)

    logger.debug(sig_gen.indicators)

    for n, ind in enumerate(sig_gen.indicators):
        logger.debug(
            "[%s] %s-> %s -> %s (%s)",
            n, ind, ind.parameters, ind.parameter_space, type(ind)
            )
        logger.debug(ind.__dict__)
        methods = [m for m in dir(ind) if '_' not in m]
        logger.debug(methods)
        logger.debug("-" * 160)


def test_get_all_operands():
    """Gets all operands in the SignalGenerator. """
    sig_gen = test_factory(linreg_roc_btc_1d)

    for n, cond in enumerate(sig_gen.conditions):
        for op in filter(lambda x: x is not None, cond.operands):
            logger.debug("[%s] %s", n, op)
            logger.debug("-" * 160)


def test_plot_desc(sig_gen):
    pprint(sig_gen.plot_desc)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    sig_gen = test_factory(sig_def)

    # sig_gen = test_factory_from_existing(sig_gen)

    # test_signal_definition(True)

    # sig_gen = test_factory(linreg_roc)
    # test_plot_desc(sig_gen)

    # test_get_all_used_indicators()
    # test_get_all_operands()

    if not sig_gen:
        sys.exit()

    # test_plot_desc(sig_gen)
    test_plot(sig_gen, data)
    # test_returns(sig_gen, data, True)

    sys.exit()

    runs = 1_000
    data = data
    st = time.time()

    logger.setLevel(logging.ERROR)

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = sig_gen.execute(data)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)

    )

    print(f'length data: {len(data["close"])} periods')
    print(f"execution time: {((time.time() - st)*1_000_000/runs):.2f} microseconds")