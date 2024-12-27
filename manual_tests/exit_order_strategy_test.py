#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
from email.quoprimime import quote
import sys
import os
import time
import logging
import pandas as pd

from typing import Iterable
from pprint import pprint


LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.staff.hermes import Hermes
import analysis.strategies.exit_order_strategies as es
from analysis.strategies.exit_order_strategies import *
from util.timeops import execution_time


# prepare an OHLCV dataframe
h = Hermes(exchange='kucoin', mode='live')
res = h.get_ohlcv(symbols='BTC-USDT', interval='1h', start=-1000, end='now UTC')
if res['success']:
    df = res['message']\
        .drop(
            ['open time', 'close time', 'volume', 'quote asset volume']
              , axis=1
        )
else:
    logger.error(res)
    df = None

# instantiate strategies
sl_atr = AtrStopLossStrategy(params=dict(fractions=1))
tp_atr = AtrTakeProfitStrategy(params=dict(fractions=2))


# =============================================================================
def test_add_stop_loss_prices_to_df(runs=100):
    if df is None:
        raise ValueError('df must not be None!')

    sl_atr.add_stop_loss_prices_to_df(df)
    st = time.time()

    for _ in range(runs):
        res = sl_atr.add_stop_loss_prices_to_df(df)

    et = round((time.time() - st) / 50 * 1_000_000, 2)
    print(f'avg time for add_stop_loss_prices_to_df: {et}µs')

    return res # type: ignore

def test_get_stop_prices():
    s =AtrStopLossStrategy(params=dict(fractions=2))
    for _ in range(5):
        res = s.get_stop_loss_prices(df, long_or_short='long')

    return res # type: ignore


def test_add_take_profit_prices_to_df():
    s = AtrTakeProfitStrategy(params=dict(fractions=3))
    for _ in range(5):
        res = s.add_take_profit_prices_to_df(df)

    return res # type: ignore

def test_get_take_profit_prices():
    s =AtrTakeProfitStrategy(params=dict(fractions=2))
    for _ in range(5):
        res = s.get_take_profit_prices(df, long_or_short='long')

    return res # type: ignore

def test_get_sl_prices_np( strat: IExitOrderStrategy):
    if df is None:
        raise ValueError('df must not be None!')

    kwargs = {
        'open_': df.open.to_numpy(),
        'high': df.high.to_numpy(),
        'low': df.low.to_numpy(),
        'close': df.close.to_numpy()
    }

    st = time.time()

    for _ in range(50):
        res = strat.get_trigger_prices_np(**kwargs)

    et = round((time.time() - st) / 50 * 1_000_000, 2)
    print(f'avg time for get_trigger_prices_np: {et}µs')

    df['sl_long'], df['sl_short'] = res[0], res[1]

    return df


def test_get_valid():
    pprint(es.get_valid_strategies())

def test_sl_strategy_factory():
    strategies = ('atr', 'percent', 'bogus')
    atr_params = {'atr_lookback': 14, 'atr_factor': 5, 'is_trailing': True}
    percent_params = {'percent': 10, 'is_trailing': False, 'bogus': None}

    strategy = 'atr' # choice(strategies)

    if strategy == 'atr':
        sl_def = StopLossDefinition(strategy=strategy, params=atr_params)
        try:
            res = sl_strategy_factory(sl_def)
        except Exception as e:
            logger.error(e)

        assert isinstance(res, IStopLossStrategy)

        for param, value in atr_params.items():
            assert getattr(res, param) == value


    elif strategy == 'percent':
        sl_def = StopLossDefinition(strategy=strategy, params=percent_params)
        try:
            res = sl_strategy_factory(sl_def)
        except Exception as e:
            logger.error(e)

        assert isinstance(res, IStopLossStrategy)

        # for param, value in percent_params.items():
        #     assert getattr(res, param) == value


    else:
        sl_def = StopLossDefinition(strategy=strategy, params=None)
        try:
            res = sl_strategy_factory(sl_def)
        except Exception as e:
            logger.error(e)
            res = None

    if res:
        logger.info(res)


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    if df is not None:
        # print(test_add_stop_loss_prices_to_df())
        # print(test_get_stop_prices())

        #print(test_add_take_profit_prices_to_df())
        # print(test_get_take_profit_prices())

        # print(df)

        # print(test_get_sl_prices_np(sl_atr))

        test_get_valid()
        # test_sl_strategy_factory()
