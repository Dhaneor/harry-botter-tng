#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:36:53 2023

@author: dhaneor
"""
import sys
import os
import pandas as pd

# -------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -------------------------------------------------------------------------------------

from src.staff.hermes import Hermes  # noqa: E402, F401
from src.broker.models.symbol import Symbol  # noqa: E402, F401


hermes = Hermes(exchange='binance', verbose=True)


# -------------------------------------------------------------------------------------
def get_sample_data(length: int, interval: str = '15min') -> dict:
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
            ).dropna(inplace=True)  # noqa: E123

    # df.dropna(inplace=True)

    start, end = len(df) - length, -1
    return {col: df[start:end][col].to_numpy() for col in df.columns}


def get_ohlcv(symbol: str, interval: str, as_dataframe: bool = False) -> pd.DataFrame:
    """Get the 1000 most recent periods of OHLCV data for a symbol.

    Parameters
    ----------
    symbol : str
        symbol_name
    interval : str
        trading interval

    Returns
    -------
    pd.DataFrame
        OHLCV dataframe

    Raises
    ------
    Exception
        if the data could not be found
    """

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=-2000, end='now UTC'
    )

    if not res.get('success'):
        raise Exception(res.get('error'))

    ohlcv = res.get('message')

    drop_cols = [
        c for c
        in ['quote asset volume', 'close time']
        if c in ohlcv.columns
    ]

    ohlcv.drop(drop_cols, inplace=True, axis=1)

    if as_dataframe:
        return ohlcv

    return {col: ohlcv[col].to_numpy() for col in ohlcv.columns}


def get_symbol(self, symbol_name: str) -> Symbol:
    s = hermes.get_symbol(symbol_name)
    return self.symbol_factory.build_from_database_response(s)


if __name__ == "__main__":
    print(get_sample_data(50))
