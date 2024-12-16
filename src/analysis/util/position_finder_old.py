#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28 17:48:20 2023

@author dhaneor
"""
import logging
import pandas as pd
import numpy as np

from numba import jit, int8
from typing import Optional, Tuple

from util.timeops import execution_time

LOG_LEVEL = "INFO"
logger = logging.getLogger('main.position_finder')
logger.setLevel(LOG_LEVEL)

MAX_LEVERAGE = 20


# =============================================================================
@execution_time
def find_positions(df: pd.DataFrame) -> pd.DataFrame:

    if 's.all' in df.columns:
        df.rename({'s.all': 'signal'}, axis=1, inplace=True)

    try:
        human_open_time = df['human open time'].copy()
        close_time = df['close time'].copy()
        df.drop(columns=['human open time', 'close time'], inplace=True)
    except:
        logger.debug('could not find human open time or close time')
        logger.debug(df.columns.to_list())

    columns = tuple(df.columns.to_list())

    numeric_cols = [col for col in columns if df[col].dtype == np.float64]
    bool_cols = [col for col in columns if df[col].dtype == np.bool_]

    for col in bool_cols:
        print(f'converting {col} to numeric type')
        df[col] = df[col].replace(np.nan, 0).astype(np.int8)

    print(df.info())

    data = df.to_numpy()
    data = find_positions_nb(data, columns)

    df = pd.DataFrame(data=data, columns=columns)

    for col in numeric_cols:
        try:
            df[col] = df[col].astype(np.float64)
        except Exception as e:
            logger.debug(e)

    df['sl_trig'] = df['sl_trig'].replace(np.nan, False).astype(bool)

    return df


@execution_time
@jit(nopython=True, cache=True)
def find_positions_nb(data, columns):

    num_rows = data.shape[0]
    active_position = int8(0)
    buy_signal = int8(0)
    sell_signal = int8(0)
    close_long_signal = int8(0)
    close_short_signal = int8(0)

    open_ = columns.index('open')
    close = columns.index('close')
    high = columns.index('high')
    low = columns.index('low')
    signal = columns.index('signal')
    pos = columns.index('position')
    sl_long = columns.index('sl_long')
    sl_short = columns.index('sl_short')
    sl_current = columns.index('sl_current')
    sl_trig = columns.index('sl_trig')
    buy = columns.index('buy')
    sell = columns.index('sell')
    buy_at = columns.index('buy_at')
    sell_at = columns.index('sell_at')

    # ..........................................................................
    def _open_long_position(row: np.ndarray) -> np.ndarray:
        row[pos] = 1
        row[buy] = 1
        row[buy_at] = row[open_]
        row[sl_current] = row[sl_long]
        return row

    def _continue_long_position(row: np.ndarray) -> np.ndarray:
        row[pos] = 1
        row[sl_current] = max(row[sl_long], row[sl_current])
        return row

    def _close_long_position(row: np.ndarray) -> np.ndarray:
        row[sell] = 1
        row[sell_at] = row[close]

        return row

    def _open_short_position(row: np.ndarray) -> np.ndarray:
        row[pos] = -1
        row[sell] = 1
        row[sell_at] = row[open_]
        row[sl_current] = row[sl_short]
        return row

    def _continue_short_position(row: np.ndarray) -> np.ndarray:
        row[pos] = -1
        row[sl_current] = min(row[sl_short], row[sl_current])
        return row

    def _close_short_position(row: np.ndarray) -> np.ndarray:
        row[buy] = 1
        row[buy_at] = row[close]
        return row

    # ..........................................................................
    # main body of the function

    # iterate through rows and adjust position values over time
    # path-dependant -> needs to be a loop
    for idx in range(1, num_rows):

        row = data[idx]

        row[pos] = 0
        row[sl_trig] = 0

        buy_signal = 1 if data[idx - 1][signal] == 1 else 0
        sell_signal = 1 if data[idx - 1][signal] == -1 else 0
        close_long_signal = 1 if row[signal] in (0, -1) else 0
        close_short_signal = 1 if row[signal] in (0, 1) else 0

        # continue LONG position
        if active_position == 1:
            row[sl_current] = data[idx - 1][sl_current]
            row = _continue_long_position(row)
            active_position = 1

        # open LONG position
        if active_position != 1 and buy_signal:
            row = _open_long_position(row)
            active_position = 1

        # close LONG position
        if active_position == 1:
            if close_long_signal:
                row = _close_long_position(row)
                active_position = 0

            if (row[low] < row[sl_current]):
                row[sl_trig] = 1
                row[sell] = 1
                row[sell_at] = row[sl_current]
                active_position = 0

        # ......................................................................
        # continue SHORT position
        if active_position == -1:
            row[sl_current] = data[idx - 1][sl_current]
            row = _continue_short_position(row)
            active_position = -1

        # open SHORT position
        if active_position != -1 and sell_signal:
            row = _open_short_position(row)
            active_position = -1

        # close SHORT position
        if active_position == -1:
            if close_short_signal:
                row = _close_short_position(row)
                active_position = 0
            if (row[high] > row[sl_current]):
                row[sl_trig] = 1
                row[buy] = 1
                row[buy_at] = row[sl_current]
                active_position = 0

        data[idx] = row

    return data




# =============================================================================
@execution_time
def process_position(df: pd.DataFrame) -> pd.DataFrame:
    res = process_position_nb(
        df.open.to_numpy(),
        df.close.to_numpy(),
        df.high.to_numpy(),
        df.low.to_numpy(),
        df.position.to_numpy(),
        df['s.all'].to_numpy(),
        df['sl_long'].to_numpy(),
        df['sl_short'].to_numpy(),
        df['sl_current'].to_numpy(),
        df['sl_trig'].astype(float).to_numpy(),
        df.buy.replace('•', 1).replace('', 0).astype(float).to_numpy(),
        df.sell.replace('•', 1).replace('', 0).astype(float).to_numpy(),
        df['buy.at'].to_numpy(),
        df['sell.at'].to_numpy(),
    )

    df['position'] = res[0]
    df['sl_current'] = res[5]
    df['sl_trig'] = res[6].astype(bool)
    df['buy'] = res[1]
    df['sell'] = res[3]
    df['buy.at'] = res[2]
    df['sell.at'] = res[4]

    df['buy'].replace(1, '•', inplace=True)
    df['sell'].replace(1, '•', inplace=True)
    # df['buy'].replace(0, '', inplace=True)
    # df['sell'].replace(0, '', inplace=True)

    return df

@execution_time
@jit(nopython=True, cache=True)
def process_position_nb(open_: np.ndarray, close: np.ndarray,
                        high: np.ndarray, low: np.ndarray,
                        position: np.ndarray, signal: np.ndarray,
                        sl_long: np.ndarray, sl_short: np.ndarray,
                        sl_current: np.ndarray, sl_trig: np.ndarray,
                        buy: np.ndarray, sell: np.ndarray,
                        buy_at: np.ndarray, sell_at: np.ndarray
                        ) -> Tuple[np.ndarray, ...]:

    num_rows = open_.shape[0]
    active_position = int8(0)
    prev_sl_triggered = int8(0)
    buy_signal = int8(0)
    sell_signal = int8(0)

    for idx in range(num_rows):
        sl_trig[idx] = False

        if position[0] > 0:
            close_signal = True if signal[idx] in (0, -1) else False

            if idx == 0:
                buy[idx] = 1
                buy_at[idx] = open_[idx]
                sl_current[idx] = sl_long[idx]
                buy_signal = True

            else:
                prev_sl_triggered = sl_trig[idx-1]
                active_position = (position[idx-1] == 1) and (not prev_sl_triggered)
                buy_signal = (signal[idx-1] == 1)
                position[idx] = 0

            if active_position and (not prev_sl_triggered):
                position[idx] = 1
                buy[idx] = 0
                buy_at[idx] = np.nan
                sl_current[idx] = max(sl_current[idx-1], sl_long[idx])

            if (not active_position) and buy_signal:
                position[idx] = 1
                buy[idx] = 1
                buy_at[idx] = open_[idx]
                sl_current[idx] = sl_long[idx]

            if (active_position or buy_signal) and close_signal:
                position[idx] = 1
                sell[idx] = 1
                sell_at[idx] = close[idx]

            sl_trig[idx] = low[idx] < sl_current[idx]

            if (position[idx] == 1) and sl_trig[idx]:
                sell[idx] = 1
                sell_at[idx] = sl_current[idx]

        elif position[0] < 0:
            close_signal = True if signal[idx] in (0, 1) else False

            if idx == 0:
                sell[idx] = 1
                sell_at[idx] = open_[idx]
                sl_current[idx] = sl_short[idx]
                sell_signal = True

            else:
                prev_sl_triggered = sl_trig[idx - 1]
                active_position = (position[idx - 1] == -1) and (not prev_sl_triggered)
                sell_signal = (signal[idx - 1] == -1)
                position[idx] = 0

            if active_position and (not prev_sl_triggered):
                position[idx] = -1
                sell[idx] = 0
                sell_at[idx] = np.nan
                sl_current[idx] = min(sl_current[idx - 1], sl_short[idx])

            if (not active_position) and sell_signal:
                position[idx] = -1
                sell[idx] = 1
                sell_at[idx] = open_[idx]
                sl_current[idx] = sl_short[idx]

            if (active_position or sell_signal) and close_signal:
                position[idx] = -1
                buy[idx] = 1
                buy_at[idx] = close[idx]

            sl_trig[idx] = high[idx] > sl_current[idx]

            if (position[idx] == -1) and sl_trig[idx]:
                buy[idx] = 1
                buy_at[idx] = sl_current[idx]

        else:
            sl_long, sl_short = np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    return position, buy, buy_at, sell, sell_at, sl_current, sl_trig


# =============================================================================
class PositionFinder:

    def add_event_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df['event.id'] = (np.random.rand(len(df)) * 1_000_000).astype(int)
        df.loc[(df['position'] == 0) & (df['position'].shift() == 0), 'event.id'] = np.nan
        df.loc[(df['position'] == df['position'].shift()), 'event.id'] = np.nan
        df['event.id'].ffill(inplace=True)

        return df

    # ------------------------------------------------------------------------------
    @classmethod
    def show_overview(
        df: pd.DataFrame,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ):

        include_columns = [
            'human open time', 'open', 'high', 'low', 'close', 'signal', 's.bo',
            'position', 'event.id', 'leverage',
            # 'sl_long', 'sl_short'
        ]

        for c in df.columns:
            if c.split('.')[0] == 'p':
                include_columns.append(c)
            if c.split('_')[0] == 'buy':
                include_columns.append(c)
            if c.split('_')[0] == 'sell':
                include_columns.append(c)
            if c.split('.')[0] == 'tp':
                include_columns.append(c)
            if c.split('.')[0] == 'b':
                include_columns.append(c)

        stop_loss_columns = ['sl_current', 'sl.pct', 'sl_trig', 'sl.l.trig']

        for col in stop_loss_columns:
            if col in df.columns:
                include_columns.append(col)

                if col == 'sl.pct':
                    df['sl.pct'] = df['sl.pct'] * 100

        include_columns.append('returns.log')
        include_columns.append('s.returns')

        # .....................................................................
        # set pandas display options

        # pd.set_option('precision', 8)
        pd.options.display.max_rows = 400
        pd.set_option("display.max_rows", 400)
        pd.set_option("display.min_rows", 100)
        # df['b.base'] = df['b.base'].apply(lambda x: '%.8f' % x)

        # replace certain values for readability
        df = df.replace(np.nan, '.', regex=True)
        df = df.replace(False, '', regex=True)
        df = df.replace(0, '', regex=True)

        # make sure display columns are available in dataframe
        include_columns = [col for col in include_columns if col in df.columns]

        if not start_index:
            start_index = df.index.values[0]
        if not end_index:
            end_index = df.index.values[-1]

        print('=' * 200)
        print(df.loc[start_index:end_index, include_columns])
        print(df.info())
