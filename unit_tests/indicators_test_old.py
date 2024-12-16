#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import sys
import os
import pandas as pd
from time import time
from pprint import pprint

# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

# -----------------------------------------------------------------------------
from src.staff.hermes import Hermes
from src.analysis.indicators import Indicators
from util.timeops import execution_time

SYMBOL = 'BTCUSDT'
INTERVAL = '15m'
START = -100000 # 'January 01, 2018 00:00:00'
END = 'January 01, 2023 00:00:00'

HERMES = Hermes('binance')
INDICATORS = Indicators()

PARAMS = {

    'sma' : (
        INDICATORS.sma, {'period' : 21}
    ),
    'ewma' : (
        INDICATORS.ewma, {'period' : 21}
    ),
    'atr' : (
        INDICATORS.average_true_range, {'period' : 14}
    ),
    'rsi' : (
        INDICATORS.rsi,
        {'on_what' : 'close', 'lookback' : 14, 'column' : None}
    ),
    'droc' : (
        INDICATORS.dynamic_rate_of_change,
        {'on_what' : 'close', 'smoothing' : 3}
    ),
    'disp' : (
        INDICATORS.disparity_index, {}
    ),
    'streak' : (
        INDICATORS.streak_duration, {}
    ),
    'c_rsi' : (
        INDICATORS.connors_rsi, {}
    ),
    'noise_index' : (
        INDICATORS.noise_index, {}
    ),
    'fibonacci_trend' : (
        INDICATORS.fibonacci_trend_nb, {'max_no_of_periods' : 100}
    ),
    'trendy_index' : (
        INDICATORS.trendy_index, {'lookback' : 14}
    ),
    'keltner_talib' : (
        INDICATORS.keltner_talib, {'kc_lookback' : 20}
    ),
    'bollinger' : (
        INDICATORS.bollinger, {'period' : 20, 'multiplier' : 2}
    ),

}

# ==============================================================================
def get_data():
    res = HERMES.get_ohlcv(
        symbols=SYMBOL, interval=INTERVAL, start=START, end=END
        )

    df: pd.DataFrame

    if res.get('success'):
        df = res.get('message') # type: ignore
    else:
        pprint(res)
        df = pd.DataFrame()

    if res.get('success') and not df.empty:
        drop_cols = [
            'open time', 'volume', 'close time', 'quote asset volume'
        ]
        df.drop(columns=drop_cols, inplace=True)
        return df
    else:
        return pd.DataFrame()

# ------------------------------------------------------------------------------
@execution_time
def test_indicator(data: pd.DataFrame, name: str):
    method = PARAMS[name][0]
    kwargs = PARAMS[name][1]
    kwargs['df'] = data

    df = None

    for _ in range(3):
        df = method(**kwargs)

    if df is not None:
        print(df.tail(10))


@execution_time
def test_indicator_with_series(df: pd.DataFrame, name: str):
    method = PARAMS[name][0]
    kwargs = PARAMS[name][1]
    kwargs['data'] = data.close

    for _ in range(3):
        df[name] = method(**kwargs)
    print(df.tail(30))

@execution_time
def test_bollinger(df: pd.DataFrame):
    data['bb.mid'], data['bb.upper'], data['bb.lower'] = INDICATORS.bollinger(
        data.close.to_numpy()
    )
    print(df.tail(30))

# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    data = get_data()

    # test_indicator(data, 'noise_index')
    # test_indicator_with_series(data, 'noise_index')

    test_bollinger(data)




