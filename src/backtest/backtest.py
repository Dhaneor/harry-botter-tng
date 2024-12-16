#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 06 21:12:20 2023

@author dhaneor
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from numba import jit, int8

logger = logging.getLogger('main.backtest')
logger.setLevel('DEBUG')

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------

from src.staff.hermes import Hermes  # noqa: E402, F401
from src.analysis.oracle import Oracle  # noqa: E402, F401
from src.models.symbol import Symbol  # noqa: E402, F401
from src.plotting.minerva import BacktestChart   # noqa: E402, F401
from util.timeops import execution_time   # noqa: E402, F401

trade_costs = 0.002


@jit(nopython=True, cache=True)
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
        target_exposure = (b_quote[i] + b_base[i] * close[i-1]) * leverage[i]
        target_size = target_exposure / close[i-1]
        change_size = target_size - b_base[i]
        change_pct = abs(change_size / b_base[i])

        if change_pct >= min_change_pct:
            if increase_allowed and change_size > 0:
                buy_at[i] = close[i-1]
                buy_size[i] = change_size
                b_base[i] = b_base[i] + (buy_size[i] * (1 - trade_costs))
                b_quote[i] = b_quote[i] - (buy_size[i] * close[i-1])
            elif decrease_allowed and change_size < 0:
                sell_at[i] = close[i-1]
                sell_size[i] = abs(change_size)
                b_base[i] = b_base[i] - sell_size[i]
                b_quote[i] = b_quote[i] + ((sell_size[i] * close[i-1]) * (1 - trade_costs))

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
        size = budget / sell_at[i] * (1- trade_costs)

        b_base[i] = b_base[i] - size
        b_quote[i] = b_quote[i] + budget

    # closing SHORT position
    if buy_at[i] > 0:
        quote_spent = abs(b_base[i]) * buy_at[i] * (1 + trade_costs)
        b_quote[i] = b_quote[i] - quote_spent
        b_base[i] = 0
        return b_base, b_quote

    # increase/decrease LONG position if necessary
    if i > 0 and continuous == 1 and not sell_at[i] > 0:
        target_exposure = (b_quote[i] + b_base[i] * close[i-1]) * leverage[i]
        target_size = (target_exposure / close[i-1]) * -1
        change_size = target_size - b_base[i]
        change_pct = abs(change_size / b_base[i])

        if change_pct >= min_change_pct:
            if decrease_allowed and change_size > 0:
                buy_at[i] = close[i-1]
                buy_size[i] = change_size
                b_base[i] = b_base[i] + buy_size[i]
                b_quote[i] = b_quote[i] - ((buy_size[i] * close[i-1]) * (1 - trade_costs))
            elif increase_allowed and change_size < 0:
                sell_at[i] = close[i-1]
                sell_size[i] = abs(change_size)
                b_base[i] = b_base[i] - sell_size[i]
                b_quote[i] = b_quote[i] + ((sell_size[i] * close[i-1]) * (1 - trade_costs))

    return b_base, b_quote


# ==============================================================================
class Backtest:

    def __init__(self, exchange: str):
        self.hermes = Hermes(exchange=exchange, mode='backtest', verbose=True)
        self.oracle = Oracle()
        self.trade_costs = 0.002

    @execution_time
    def run(self, symbol: str, interval: str, strategy: str,
            start: Union[str, int], end: Union[str, int],
            leverage: float, risk_level: int, strategy_params: dict,
            stop_loss_strategy: str, stop_loss_params: dict):

        # ......................................................................
        # load OHLCV data
        ohlcv = self._get_ohlcv(symbol, interval, start, end)

        if ohlcv is None:
            logger.error(f'No OHLCV data found for {symbol}')
            return

        if not isinstance(ohlcv, pd.DataFrame):
            logger.error(f'OHLCV data is not a dataframe: {ohlcv}')
            return

        # ......................................................................
        # add signals and (theoretical) positions to OHLCV dataframe
        self.oracle.set_strategy(strategy)

        if stop_loss_strategy:
            self.oracle.set_sl_strategy(stop_loss_params)

        # ohlcv = self.oracle.speak(
        #     data=ohlcv,
        #     strategy=strategy,
        #     strategy_params=strategy_params,
        #     sl_strategy=stop_loss_strategy,
        #     sl_params=stop_loss_params,
        #     risk_level=risk_level,
        #     max_leverage=leverage
        # )

        ohlcv = self.oracle.speak(ohlcv)

        if ohlcv is None:
            logger.error(f'No positions found for {symbol}')
            return

        # when loading data from Hermes, the first 200 rows of the
        # dataframe are just a padding. This helps with calculating
        # different indicators (like moving averages), but is not
        # necessary or wanted for backtesting -> remove them
        ohlcv = ohlcv.iloc[200:, :].copy(deep=True).reset_index()

        # ......................................................................
        # calculate the actual trades
        ohlcv['b.base'] = np.nan
        ohlcv['b.quote'] = np.nan
        ohlcv['b.value'] = np.nan

        ohlcv.at[0, 'b.base'] = 0
        ohlcv.at[0, 'b.quote'] = strategy_params.get('initial_capital')

        print(ohlcv)

        if ohlcv.at[0, 'position'] == 1:
            ohlcv.at[0, 'buy'] = True
            ohlcv.at[0, 'buy_at'] = ohlcv.at[0, 'open']

        elif ohlcv.at[0, 'position'] == -1:
            ohlcv.at[0, 'sell'] = True
            ohlcv.at[0, 'sell_at'] = ohlcv.at[0, 'open']

        ohlcv = self.calculate_trades_fast(ohlcv)

        # ......................................................................
        # calculate the value of the account/portfolio
        ohlcv['b.value'] = ohlcv['b.quote'] + ohlcv['b.base'] * ohlcv['close']

        # replace numerical values with strings for 'position'
        # for better readability
        conditions = [
            (ohlcv['position'] == 1),
            (ohlcv['position'] == -1),
        ]
        choices = ['LONG', 'SHORT']
        ohlcv['position'] = np.select(conditions, choices, default='')

        return ohlcv

    # --------------------------------------------------------------------------
    def _get_symbol(self, symbol: str):
        try:
            res = self.hermes.get_symbol(symbol)
        except Exception as e:
            logger.exception(e)
            return None
        else:
            logger.info(f'Symbol: {res}')
            return res

    @execution_time
    def _get_ohlcv(self, symbol: str, interval: str,
                   start: Union[str, int], end: Union[str, int]
                   ) -> Union[pd.DataFrame, None]:
        try:
            res = self.hermes.get_ohlcv(symbol, interval, start, end)
        except Exception as e:
            logger.exception(e)
            return

        if res.get('success'):
            return res.get('message', None)
        else:
            logger.error(res.get('error', None))
            return None

    @execution_time
    def calculate_trades_fast(self, df: pd.DataFrame) -> pd.DataFrame:

        df['b.base'], df['b.quote'] = calculate_trades_nb(
            close=df['close'].to_numpy(),
            position=df['position'].to_numpy(),
            buy_at=df['buy_at'].to_numpy(),
            sell_at=df['sell_at'].to_numpy(),
            buy_size=df['buy_size'].to_numpy(),
            sell_size=df['sell_size'].to_numpy(),
            leverage=df['leverage'].to_numpy(),
            b_base=df['b.base'].to_numpy(),
            b_quote=df['b.quote'].to_numpy()
        )

        return df

    # --------------------------------------------------------------------------
    def show_overview(
        self,
        df: pd.DataFrame,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ):

        include_columns = [
            'human open time', 'close',
            'signal', 'position', 'leverage',
        ]

        stop_loss_columns = ['sl_current', 'sl_pct', 'sl_trig']

        for col in stop_loss_columns:
            if col in df.columns:
                include_columns.append(col)

                if col == 'sl.pct':
                    df['sl.pct'] = df['sl.pct'] * 100

        for c in df.columns:
            if c.split('.')[0] == 'p': include_columns.append(c)
            if c.split('_')[0] == 'buy': include_columns.append(c)
            if c.split('_')[0] == 'sell': include_columns.append(c)
            if c.split('_')[0] == 'tp': include_columns.append(c)

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