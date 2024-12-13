#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 06 21:12:20 2023

@author dhaneor
"""
import pandas as pd
import numpy as np
from numba import jit, int8


@jit(nopython=True)
def merge_signals_nb(open_long, open_short, close_long, close_short):
    """Merges the four possible signals into one column."""
    n = len(open_long)
    signal = np.zeros(n, dtype=np.float64)
    position = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            if open_long[i] > 0:
                signal[i] = open_long[i]
                position[i] = 1
            elif open_short[i] > 0:
                signal[i] = open_short[i] * -1
                position[i] = -1
        else:
            prev_position = position[i-1]
            signal[i] = signal[i-1]
            position[i] = prev_position

            if open_long[i] > 0:
                signal[i] = open_long[i]
                position[i] = 1

            elif close_long[i] > 0:
                if prev_position > 0:
                    signal[i] = 0
                    position[i] = 0

            elif open_short[i] > 0:
                signal[i] = open_short[i] * -1
                position[i] = -1

            elif close_short[i] > 0:
                if prev_position < 0:
                    signal[i] = 0
                    position[i] = 0

    return signal, position


def merge_signals(data):
    open_long = np.nan_to_num(data["open_long"])
    open_short = np.nan_to_num(data["open_short"])
    close_long = np.nan_to_num(data["close_long"])
    close_short = np.nan_to_num(data["close_short"])

    signal, position = merge_signals_nb(open_long, open_short, close_long, close_short)

    data["signal"] = signal
    data["position"] = position
    return data


def find_positions(df: pd.DataFrame) -> pd.DataFrame:

    find_positions_nb(
        df.open.to_numpy(),
        df.high.to_numpy(),
        df.low.to_numpy(),
        df.close.to_numpy(),
        df.signal.to_numpy(),
        df.position.to_numpy(),
        df.buy.to_numpy(),
        df.buy_at.to_numpy(),
        df.sell.to_numpy(),
        df.sell_at.to_numpy(),
        df.sl_long.to_numpy(),
        df.sl_short.to_numpy(),
        df.sl_current.to_numpy(),
        df.sl_trig.to_numpy(),
        df.tp_long.to_numpy(),
        df.tp_short.to_numpy(),
        df.tp_current.to_numpy()
    )

    df.sl_current.replace(0, np.nan, inplace=True)
    df.buy = df.buy.astype(bool)
    df.sell = df.sell.astype(bool)
    df.sl_trig = df.sl_trig.astype(bool)

    return df


def find_positions_with_dict(data: dict) -> dict:

    if "signal" not in data:
        merge_signals(data)

    for col in (
        "position",
        "buy", "buy_size", "buy_at",
        "sell", "sell_size", "sell_at",
        "sl_long", "sl_short", "sl_current", "sl_trig",
        "tp_long", "tp_short", "tp_current"
    ):
        if col not in data:
            if col in ("buy_size", "sell_size"):
                data[col] = np.zeros(data["close"].shape[0])
            else:
                data[col] = np.full(data["close"].shape[0], np.nan)

    res = find_positions_nb(
        data["open"],
        data["high"],
        data["low"],
        data["close"],
        data["signal"],
        data["position"],
        data["buy"],
        data["buy_at"],
        data["sell"],
        data["sell_at"],
        data["sl_long"],
        data["sl_short"],
        data["sl_current"],
        data["sl_trig"],
        data["tp_long"],
        data["tp_short"],
        data["tp_current"]
    )

    data["position"] = res[0]
    data["buy"] = res[1]
    data["buy_at"] = res[2]
    data["sell"] = res[3]
    data["sell_at"] = res[4]
    data["sl_current"] = res[5]
    data["sl_trig"] = res[6]

    return data


@jit(nopython=True, cache=True)
def find_positions_nb(open_, high, low, close, signal, position,
                      buy, buy_at, sell, sell_at,
                      sl_long, sl_short, sl_current, sl_trig,
                      tp_long, tp_short, tp_current):

    num_rows = open_.shape[0]
    active_position = int8(0)

    for i in range(1, num_rows):

        position[i] = 0
        sl_trig[i] = 0

        # continue LONG position
        if active_position == 1:
            position[i] = 1
            sl_current[i] = max(sl_current[i - 1], sl_long[i - 1])

        # open LONG position
        if active_position != 1 and signal[i - 1] > 0:
            active_position = 1
            position[i] = 1
            buy[i] = 1
            buy_at[i] = close[i - 1]
            sl_current[i] = sl_long[i - 1]

        # close LONG position
        if active_position == 1:
            if signal[i] <= 0:
                active_position = 0
                sell[i] = 1
                sell_at[i] = close[i]

            if sl_current[i] and low[i] < sl_current[i]:
                active_position = 0
                sl_trig[i] = 1
                sell[i] = 1
                sell_at[i] = sl_current[i]

        # continue SHORT position
        if active_position == -1:
            position[i] = -1
            sl_current[i] = min(sl_current[i - 1], sl_short[i - 1])

        # open SHORT position
        if active_position != -1 and signal[i - 1] < 0:
            active_position = -1
            position[i] = -1
            sell[i] = 1
            sell_at[i] = close[i - 1]
            sl_current[i] = sl_short[i - 1]

        # close SHORT position
        if active_position == -1:
            if signal[i] >= 0:
                active_position = 0
                buy[i] = 1
                buy_at[i] = close[i]

            if sl_current[i] and high[i] > sl_current[i]:
                active_position = 0
                sl_trig[i] = 1
                buy[i] = 1
                buy_at[i] = sl_current[i]

    return position, buy, buy_at, sell, sell_at, sl_current, sl_trig
