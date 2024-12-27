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
import numpy as np
from pprint import pprint
from typing import Union, Iterable, Optional

# profiler imports
from cProfile import Profile
from pstats import Stats

# configure logger
LOG_LEVEL = logging.ERROR
logger = logging.getLogger('main')
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s.%(funcName)s  - [%(levelname)s]: %(message)s'
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.analysis.oracle import Oracle, atr_nb  # noqa: E402, F401
from src.staff.hermes import Hermes  # noqa: E402, F401
from analysis.indicators.indicators_old import Indicators  # noqa: E402, F401
from util.timeops import execution_time, seconds_to  # noqa: E402, F401


# =============================================================================
class OracleTester:

    def __init__(self):
        self.indicators = Indicators()
        self.oracle = Oracle()

    def get_available_strategies(self) -> list:

        return self.oracle.get_available_strategies()

    def set_strategy(self, strategy: str):
        self.oracle.set_strategy(strategy)

    def set_sl_strategy(self, sl_params: Optional[dict] = None) -> None:
        self.oracle.set_sl_strategy(sl_params)

    def speak(self, df: pd.DataFrame):
        return self.oracle.speak(data=df)

    def show_result(self, df: pd.DataFrame):

        print(f'Dataframe with signal for {symbol} ({interval})')
        print('')

        df_print = df.copy(deep=True).replace(np.nan, '-', inplace=False)
        df_print.replace(False, '', inplace=True)
        df = df.replace(0, '•', regex=True, inplace=True)
        # df = df.replace(np.nan, '.', regex=True)

        # remove columns we don't want to see
        drop_cols = [
            'open time', 'close time', 'volume', 'quote asset volume',
            'log returns', 'vol anno', 'std dev'
        ]
        drop_str = ['ewm', 'ma', 'stoch', 'kc.', 'atr', 'rsi.']
        for col in df_print.columns:
            if any(elem in col for elem in drop_str):
                drop_cols.append(col)
            if (col[:2] == 's.') and (col != 's.all'):
                drop_cols.append(col)

        drop_cols.remove('human open time')

        for col in drop_cols:
            try:
                df_print.drop(col, axis=1, inplace=True)
            except Exception as e:
                logger.error(e)

        # set display options
        pd.options.display.precision = 8
        pd.options.display.max_rows = 400
        pd.options.display.min_rows = 200

        print(df_print)

    def get_data(self, symbol: str, interval: str, start: Union[str, int],
                 end: Union[str, int]):

        exchange = 'kucoin' if '-' in symbol else 'binance'
        hermes = Hermes(exchange=exchange, mode='live')

        res = hermes.get_ohlcv(
            symbols=symbol, interval=interval, start=start, end=end
        )

        if isinstance(res, dict) and res.get('success'):
            return res.get('message')
        else:
            logger.error('unable to get OHLCV data')
            logger.error(res)
            sys.exit()


# =============================================================================
def test_calculate_atr_nb(symbol: str, interval: str, start: str | int, end: str):
    ot = OracleTester()
    df = ot.get_data(symbol=symbol, interval=interval, start=start, end=end)

    if df is not None:
        for _ in range(10):
            df['atr'] = atr_nb(
                df.open.to_numpy(),
                df.high.to_numpy(),
                df.low.to_numpy(),
            )

        print(df.tail(49))


# -----------------------------------------------------------------------------
def test_get_strategies():

    ot = OracleTester()
    strategies = ot.get_available_strategies()
    pprint(strategies)
    print('\n')


def test_set_strategy(strategy=None):
    ot = OracleTester()

    if strategy is None:
        strategies = ot.get_available_strategies()
        print('\n')
        pprint(strategies)
        print('\n')

        for s in strategies:
            res = ot.set_strategy(s)
            pprint(res)
            print('-' * 80)

    else:
        res = ot.set_strategy(strategy)
        pprint(res)
        print('-' * 80)
        pprint(ot.oracle.__dict__)


# -----------------------------------------------------------------------------
def test_oracle(ohlcv: pd.DataFrame, strategy: str,
                sl_params: Optional[dict] = None,
                draw_chart: bool = False, runs=1000):
    df = None
    exc_times = []
    res = None

    ot.set_strategy(strategy)
    ot.set_sl_strategy(sl_params)

    for _ in range(runs):
        st = time.time()
        res = ot.speak(df=ohlcv)
        exc_times.append(time.time() - st)

    try:
        df = ot.oracle._cleanup(
            pd.DataFrame.from_dict(res)
        )
        df.sl_current.replace(0, np.nan, inplace=True)

        ot.show_result(df=df)

        if draw_chart:
            ot.oracle.draw_chart(df=df.iloc[200:, :])
    except AttributeError as e:
        logger.error(
            f'Surprisingly, we do not have a dataframe, but <{type(res)}> ({e})'
        )
        logger.error(res)
    finally:
        print('\n')
        print('-' * 200)
        print(f'{ot.oracle.name} has spoken! ... {len(ohlcv)} values processed')

        # # avg_time = round(sum(exc_times)/len(exc_times) * 1000, 3)
        avg_time = seconds_to(sum(exc_times) / len(exc_times))
        print(f'avg execution time for Oracle.speak(): {avg_time}')

        avg_time = round(sum(exc_times) / len(exc_times) * 1000, 3)
        # avg_time = seconds_to(sum(exc_times)/len(exc_times))
        print(f'avg execution time for Oracle.speak(): {avg_time}')


# def test_live_oracle(ohlcv: pd.DataFrame, strategy: Optional[str],
#                 sl_strategy: Optional[str]=None, draw_chart: bool=False):

#     ot = OracleTester()
#     df = ot.get_data(symbol=symbol, interval=interval, start=start, end=end)

#     if df is None:
#         raise Exception('unable to get OHLCV data')

#     req = OracleRequest(
#         symbol=symbol, interval=interval, strategy=strategy, data=df
#     )

#     for _ in range(10):
#         df = ot.oracle.speak(req)
#         # print('='*200)
#         # print(df.tail(3))


@execution_time
def test_oracle_multi(symbols: Iterable, interval: str, start: str | int,
                      end: str):
    hermes = Hermes(exchange='kucoin')

    res = hermes.get_ohlcv(symbols=symbols, interval=interval, start=start, end=end)

    res = [elem.get('message') for elem in res if elem.get('success')]

    # pprint(res)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':

    symbol = 'ETH-USDT'
    symbols = ['BTC-USDT', 'ETH-USDT']

    interval = '1d'
    start = -750  # 'January 1, 2018 00:00:00'
    end = 'July 31, 2023 00:00:00'
    strategy = 'Pure Keltner'
    sl_params = {'strategy': 'atr', 'atr_lookback': 3}
    draw_chart = True

    df: pd.DataFrame
    ot = OracleTester()
    ohlcv = ot.get_data(symbol=symbol, interval=interval, start=start, end=end)

    if ohlcv is None or ohlcv.empty:
        raise Exception('unable to get OHLCV data')

    test_oracle(
        ohlcv=ohlcv, strategy=strategy, sl_params=sl_params, draw_chart=draw_chart
    )

    # test_oracle_multi(symbols, interval, start, end)

    # test_calculate_atr_nb( symbol, interval, start, end)

    sys.exit()

    o = Oracle()
    o.set_strategy(strategy)

    with Profile(timeunit=1000) as profile:

        for _ in range(1000):
            o.speak(ohlcv)

        (
            Stats(profile)
            .strip_dirs()
            .sort_stats('cumtime')
            # .reverse_order()
            .print_stats(30)
        )
