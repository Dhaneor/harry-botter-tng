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

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------

from src.staff.hermes import Hermes  # noqa: E402, F401
from src.analysis import strategy_builder as sb  # noqa: E402, F401
from src.analysis.util import find_positions as fp  # noqa: E402, F401
from src.analysis import strategy_backtest as bt  # noqa: E402, F401
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    contra_1, trend_1, s_tema_cross, s_breakout, s_trix, s_kama_cross,
    s_linreg,
)
from src.plotting.minerva import BacktestChart  # noqa: E402, F401
from src.backtest import result_stats as rs  # noqa: E402, F401

symbol = "BTCUSDT"
interval = "1d"

start = -2300  # 'July 01, 2020 00:00:00'
end = 'now UTC'

strategy = s_linreg
risk_level = 3
initial_capital = 10_000 if symbol.endswith('USDT') else 0.5

hermes = Hermes(exchange='kucoin', mode='backtest')
strategy = sb.build_strategy(strategy)


# ======================================================================================
def get_data(length: int = 1000):
    """
    Reads a CSV file containing OHLCV (Open, High, Low, Close, Volume)
    data for a cryptocurrency and performs data preprocessing.

    Parameters
        length
            The number of data points to retrieve. Defaults to 1000.

    Returns:
        dict
            A dictionary containing the selected columns from the
            preprocessed data as numpy arrays.
    """
    df = pd.read_csv(os.path.join(parent, "ohlcv_data", "btcusdt_15m.csv"))
    df.drop(
        ["Unnamed: 0", "close time", "quote asset volume"], axis=1, inplace=True
    )

    df['human open time'] = pd.to_datetime(df['human open time'])
    df.set_index(keys=['human open time'], inplace=True, drop=False)

    if interval != "15min":
        df = df.resample(interval)\
            .agg(
                {
                    'open time': 'min', 'human open time': 'min',
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                    'volume': 'sum'
                },
                min_periods=1
            )  # noqa: E123

    df.dropna(inplace=True)

    start = len(df) - length  # randint(0, len(df) - length)
    end = -1  # start + length
    return {col: df[start:end][col].to_numpy() for col in df.columns}


def _get_ohlcv_from_db():

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
    )

    if res.get('success'):
        df = res.get('message')
        return {col: df[col].to_numpy() for col in df.columns}

    else:
        error = res.get('error', 'no error provided in response')
        raise Exception(error)


def _run_backtest(data: dict):
    return bt.run(strategy, data, initial_capital, risk_level)


def _add_stats(df):
    rs.calculate_stats(df, initial_capital)


def _show(df):
    df.set_index(keys=['human open time'], inplace=True, drop=False)
    bt.show_overview(df=df)


# ..............................................................................
def run(data, show=False, plot=False):

    df = pd.DataFrame.from_dict(_run_backtest(data))

    _add_stats(df)

    if show:
        _show(df)

    if plot:
        df.loc[~(df['position'] == 0), 'p.actv'] = True
        df.rename(
            columns={'buy_size': 'buy.amount', 'sell_size': 'sell.amount'},
            inplace=True
        )

        chart = BacktestChart(
            df=df,  # df[200:],
            title=f'{symbol} ({interval})',
            color_scheme='day'
        )
        chart.draw()


def test_find_positions(data: dict):
    fp.find_positions_with_dict(data)

    assert "position" in data, "'position' not found in data dictionary"


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    logger.info("Starting backtest...")
    logger.info(strategy)
    run(_get_ohlcv_from_db(), True, True)

    # ..........................................................................
    sys.exit()

    logger.setLevel(logging.ERROR)
    runs = 1_000
    data_pre = [_get_ohlcv_from_db() for _ in range(runs)]
    st = time.perf_counter()

    for i in range(runs):
        # test_find_positions(data_pre[i])
        bt.run(strategy, data_pre[i], initial_capital, risk_level)

    # with Profile(timeunit=0.001) as p:
    #     for i in range(runs):
    #         bt.run(strategy, data_pre[i], initial_capital, risk_level)

    # (
    #     Stats(p)
    #     .strip_dirs()
    #     .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
    #     # .reverse_order()
    #     .print_stats(30)
    # )

    # for _ in range(runs):
    #     test_strategy_run(s, False)

    et = time.perf_counter()
    print(f'length data: {len(data_pre[0]["close"])} periods')
    print(f"execution time: {((et - st)*1_000_000/runs):.2f} microseconds")
