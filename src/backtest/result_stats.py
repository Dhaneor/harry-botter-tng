#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('main.result_stats')


# ==============================================================================
def _has_all_columns(df: pd.DataFrame):
    mandatory = ['sell', 'buy', 'b.value']
    return True if all([col for col in mandatory if col in df.columns]) else False


def _calculate_capital(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0], 'cptl.b'] = df.at[df.index[0], 'b.value']

    # update capital for each row where a position is closed
    df.loc[~(df['position'] == df['position'].shift()), 'cptl.b'] = df['b.value']

    # ffill values
    df['cptl.b'] = df['cptl.b'].ffill()

    return df


def _calculate_hodl_pnl(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:

    first_price = df.at[df.index[0], 'open']
    df['hodl.value'] = 0

    try:
        hodl_qty = round(initial_capital / first_price, 8)

    except KeyError as e:
        print(df.head(5))
        print(e)
        return df

    df['hodl.value'] = (hodl_qty * df['close']).round(8)
    return df


def _calculate_max_drawdown(df: pd.DataFrame, column: str) -> pd.DataFrame:

    if column == 'b.value':
        df['b.max'] = df[column].expanding().max()
        df['b.drawdown'] = 1 - df['b.value'] / df['b.max']
        df['b.drawdown.max'] = df['b.drawdown'].expanding().max()

    if column == 'cptl.b':
        df['cptl.max'] = df[column].expanding().max()
        df['cptl.drawdown'] = 1 - df['cptl.b'] / df['cptl.max']
        df['cptl.drawdown.max'] = df['cptl.drawdown'].expanding().max()

    if column == 'hodl.value':
        df['hodl.max'] = df[column].expanding().max()
        df['hodl.drawdown'] = 1 - df['hodl.value'] / df['hodl.max']
        df['hodl.drawdown.max'] = df['hodl.drawdown'].expanding().max()

    return df


# ------------------------------------------------------------------------------
def calculate_stats(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if not _has_all_columns(df):
        print(df.columns)
        raise ValueError('df does not have all columns')

    df = _calculate_max_drawdown(df, 'b.value')

    df = _calculate_capital(df)
    df = _calculate_max_drawdown(df, 'cptl.b')

    df = _calculate_hodl_pnl(df, initial_capital)
    df = _calculate_max_drawdown(df, 'hodl.value')

    return df
