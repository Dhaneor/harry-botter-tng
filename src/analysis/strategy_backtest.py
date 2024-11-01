#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from numba import jit, int8

from .util import proj_types as tp
from .strategy_builder import IStrategy
from .util.find_positions import find_positions_with_dict, merge_signals
from .leverage import max_leverage

logger = logging.getLogger('main.backtest')
logger.setLevel('DEBUG')

trade_costs = 0.002


# ======================================================================================
# @jit(nopython=True, cache=True)
def calculate_trades_nb(close: np.ndarray, position: np.ndarray,
                        buy_at: np.ndarray, sell_at: np.ndarray,
                        buy_size: np.ndarray, sell_size: np.ndarray,
                        leverage: np.ndarray, b_base: np.ndarray,
                        b_quote: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    continuous = int8(0)
    increase_allowed = int8(1)
    decrease_allowed = int8(1)
    min_change_pct = 0.1

    for i in range(0, len(position)):
        # copy previous values (but not for first row)
        if i > 0:
            b_base[i] = b_base[i - 1]
            b_quote[i] = b_quote[i - 1]

        leverage[i] = 1 if leverage[i] == np.nan else leverage[i]

        # process LONG position
        if position[i] == 1:
            process_long_position(
                b_base, b_quote,
                buy_at, sell_at, buy_size, sell_size,
                leverage, close, i, continuous, min_change_pct,
                increase_allowed, decrease_allowed
            )

        # process SHORT position
        if position[i] == -1:
            process_short_position(
                b_base, b_quote,
                buy_at, sell_at, buy_size, sell_size,
                leverage, close, i, continuous, min_change_pct,
                increase_allowed, decrease_allowed
            )

    return b_base, b_quote


@jit(nopython=True, cache=True)
def process_long_position(b_base: np.ndarray, b_quote: np.ndarray,
                          buy_at: np.ndarray, sell_at: np.ndarray,
                          buy_size: np.ndarray, sell_size: np.ndarray,
                          leverage: np.ndarray, close: np.ndarray,
                          i: int, continuous: int, min_change_pct: float,
                          increase_allowed: int, decrease_allowed: int
                          ) -> Tuple[np.ndarray, np.ndarray]:

    # opening LONG position
    if buy_at[i] > 0:
        budget = b_quote[i] * leverage[i]
        buy_size[i] = budget / buy_at[i] * (1 - trade_costs)

        b_base[i] = b_base[i] + buy_size[i]
        b_quote[i] = b_quote[i] - budget

    # closing LONG position after SELL signal
    if sell_at[i] > 0:
        b_quote[i] = b_quote[i] \
            + b_base[i] * sell_at[i] * (1 - trade_costs)
        b_base[i] = 0
        return b_base, b_quote

    # increase/decrease LONG position if necessary
    if i > 0 and continuous == 1 and not sell_at[i] > 0:
        target_exposure = (b_quote[i] + b_base[i] * close[i - 1]) * leverage[i]
        target_size = target_exposure / close[i - 1]
        change_size = target_size - b_base[i]
        change_pct = abs(change_size / b_base[i])

        if change_pct >= min_change_pct:
            if increase_allowed and change_size > 0:
                buy_at[i] = close[i - 1]
                buy_size[i] = change_size
                b_base[i] = b_base[i] + (buy_size[i] * (1 - trade_costs))
                b_quote[i] = b_quote[i] - (buy_size[i] * close[i - 1])
            elif decrease_allowed and change_size < 0:
                sell_at[i] = close[i - 1]
                sell_size[i] = abs(change_size)
                b_base[i] = b_base[i] - sell_size[i]
                b_quote[i] += (sell_size[i] * close[i - 1]) * (1 - trade_costs)

    return b_base, b_quote


@jit(nopython=True, cache=True)
def process_short_position(b_base: np.ndarray, b_quote: np.ndarray,
                           buy_at: np.ndarray, sell_at: np.ndarray,
                           buy_size: np.ndarray, sell_size: np.ndarray,
                           leverage: np.ndarray, close: np.ndarray,
                           i: int, continuous: int, min_change_pct: float,
                           increase_allowed: int, decrease_allowed: int
                           ) -> Tuple[np.ndarray, np.ndarray]:

    # opening SHORT position
    if sell_at[i] > 0:
        budget = b_quote[i] * leverage[i]
        size = budget / sell_at[i] * (1 - trade_costs)

        b_base[i] = b_base[i] - size
        b_quote[i] = b_quote[i] + budget

    # closing SHORT position
    if buy_at[i] > 0:
        quote_spent = abs(b_base[i]) * buy_at[i] * (1 + trade_costs)
        b_quote[i] = b_quote[i] - quote_spent
        b_base[i] = 0
        return b_base, b_quote

    # increase/decrease SHORT position if necessary
    if i > 0 and continuous == 1 and not sell_at[i] > 0:
        target_exposure = (b_quote[i] + b_base[i] * close[i - 1]) * leverage[i]
        target_size = (target_exposure / close[i - 1]) * -1
        change_size = target_size - b_base[i]
        change_pct = abs(change_size / b_base[i])

        if change_pct >= min_change_pct:
            if decrease_allowed and change_size > 0:
                buy_at[i] = close[i - 1]
                buy_size[i] = change_size
                b_base[i] = b_base[i] + buy_size[i]
                b_quote[i] -= ((buy_size[i] * close[i - 1]) * (1 - trade_costs))
            elif increase_allowed and change_size < 0:
                sell_at[i] = close[i - 1]
                sell_size[i] = abs(change_size)
                b_base[i] = b_base[i] - sell_size[i]
                b_quote[i] += ((sell_size[i] * close[i - 1]) * (1 - trade_costs))

    return b_base, b_quote


def calculate_trades(data: tp.Data, initial_capital: float = 1000) -> None:
    data['b.base'] = np.full_like(data['close'], np.nan, dtype=np.float64)
    data['b.quote'] = np.full_like(data['close'], np.nan, dtype=np.float64)
    data['b.value'] = np.full_like(data['close'], np.nan, dtype=np.float64)

    data['b.base'][0] = 0
    data['b.quote'][0] = initial_capital

    if data['position'][0] == 1:
        data['buy'][0] = True
        data['buy_at'][0] = data['open'][0]

    elif data['position'][0] == -1:
        data['sell'][0] = True
        data['sell_at'][0] = data['open'][0]

    calculate_trades_nb(
        close=data['close'],
        position=data['position'],
        buy_at=data['buy_at'],
        sell_at=data['sell_at'],
        buy_size=data['buy_size'],
        sell_size=data['sell_size'],
        leverage=data['leverage'],
        b_base=data['b.base'],
        b_quote=data['b.quote']
    )


# =====================================================================================
# @execution_time
def run(
    strategy: IStrategy,
    data: tp.Data,
    initial_capital: float,
    risk_level: float = 0
):

    # add signals
    if isinstance(strategy, IStrategy):
        strategy.speak(data)
    else:
        strategy.execute(data)
    merge_signals(data)

    # add leverage
    if risk_level:
        data["leverage"] = max_leverage(data, risk_level)
        data["leverage"] = np.clip(data["leverage"], 0, 3)
    else:
        data['leverage'] = np.full_like(data['close'], 1, dtype=np.float64)

    # remove the first 200 data points (they were only necessary for the
    # calculation of indicators and leverage)
    data = {key: arr[200:] for key, arr in data.items()}

    # add positions
    find_positions_with_dict(data)

    # calculate the actual trades
    calculate_trades(data, initial_capital)

    # # calculate the value of the account/portfolio
    data['b.value'] = np.add(
        data['b.quote'], np.multiply(data['b.base'], data['close'])
    )

    return data


# ======================================================================================
def show_overview(
    df: pd.DataFrame,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None
):

    include_columns = [
        'close', 'signal', 'position', 'leverage',
    ]

    stop_loss_columns = ['sl_current', 'sl_pct', 'sl_trig']

    for col in stop_loss_columns:
        if col in df.columns:
            include_columns.append(col)

            if col == 'sl.pct':
                df['sl.pct'] = df['sl.pct'] * 100

    for c in df.columns:
        if c.split('.')[0] == 'p':
            include_columns.append(c)
        if c.split('_')[0] == 'buy':
            include_columns.append(c)
        if c.split('_')[0] == 'sell':
            include_columns.append(c)
        if c.split('_')[0] == 'tp':
            include_columns.append(c)

    include_columns += [
        'b.base', 'b.quote', 'b.value', 'cptl.b',
        'b.drawdown.max', 'cptl.drawdown.max',
        'hodl.value', 'hodl.drawdown.max'
    ]

    # .....................................................................
    df['b.base'] = df['b.base'].apply(lambda x: '%.8f' % x)
    df['b.quote'] = df['b.quote'].apply(lambda x: '%.6f' % x)

    # replace certain values for readability
    df = df.replace(np.nan, '', regex=True)
    df = df.replace(False, '', regex=True)
    df = df.replace(0, '', regex=True)

    # replace numerical values with strings for 'position'
    # for better readability
    conditions = [(df['position'] == 1),  (df['position'] == -1)]
    choices = ['LONG', 'SHORT']
    df['position'] = np.select(conditions, choices, default='')

    # make sure display columns are available in dataframe
    include_columns = [col for col in include_columns if col in df.columns]

    if not start_index:
        start_index = df.index.values[0]
    if not end_index:
        end_index = df.index.values[-1]

    # set pandas display options

    # pd.set_option('precision', 8)
    # pd.options.display.max_rows = 400
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.min_rows", 200)

    print('=' * 200)
    print(df.loc[start_index:end_index, include_columns])
    # print(df.columns)
