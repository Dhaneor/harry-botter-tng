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
import numpy as np
import pandas as pd

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401


from staff.hermes import Hermes  # noqa: E402, F4011
from analysis import strategy_builder as sb, strategy_backtest as bt  # noqa: E402, F401
from analysis.leverage import LeverageCalculator
from analysis.util import find_positions as fp  # noqa: E402, F401
from analysis.backtest import statistics as st  # noqa: E402, F401
from analysis.strategy.definitions import (  # noqa: E402, F401
    s_tema_cross, s_breakout, s_trix, s_kama_cross,
    s_linreg, s_test_er, s_ema_multi, s_linreg_ma_cross, s_aroon_osc
)
# from plotting.minerva import BacktestChart as Chart  # noqa: E402, F401)
from analysis.chart.tikr_charts import BacktestChart as Chart  # noqa: E402, F401
from analysis.backtest import result_stats as rs  # noqa: E402, F401
from analysis.models import market_data, position_py as position  # noqa: E402, F401
from util import get_logger

logger = get_logger("main")

symbol = "BTCUSDT"
interval = "1d"

start = int(-365*3)  # 'December 01, 2018 00:00:00'
end = 'now UTC'

strategy = s_breakout
risk_level, max_leverage = 0, 1
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
    md = market_data.MarketData.from_dictionary(symbol, data)
    strategy.market_data = md
    lc = LeverageCalculator(md, risk_level, max_leverage)

    result = bt.run(
        strategy=strategy, 
        leverage_calculator=lc,
        data=data, 
        initial_capital=initial_capital
    )
    df = pd.DataFrame.from_dict(result)

    # Check if portfolio value ever goes negative
    if (df['b.value'] < 0).any():
        print("Warning: Portfolio value went below 0 during the backtest.")

        # Find the first index where portfolio value is negative
        first_negative_index = df.index[df['b.value'] < 0][0]

        # Fill specified columns with NaN from this point onwards
        columns_to_fill = ['b.base', 'b.quote', 'b.value', 'cptl.b']
        df.loc[first_negative_index:, columns_to_fill] = np.nan

        # Forward fill NaN values
        df[columns_to_fill] = df[columns_to_fill].ffill()

    return df


def _add_stats(df):
    return rs.calculate_stats(df, initial_capital)


def _show(df):
    df.set_index(keys=['human open time'], inplace=True, drop=False)
    bt.show_overview(df=df)


def display_problematic_rows(df):
    df['b.base'] = df['b.base'].astype(float)
    df['b.quote'] = df['b.quote'].astype(float)

    # Find the first index where b.base > 0
    # problem_index = df[df['b.quote'] < 0].index[0] \
    #   if any(df['b.quote'] < 0) else df.index[-1]
    problem_index = df[df['b.drawdown'] > 1].index[0] \
        if any(df['b.drawdown'] > 1) else df.index[-1]

    df = df.loc[:problem_index]
    # df = df[df['cptl.b'] == 12987.936924]

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

    # Display the relevant rows
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print('=' * 200)
    print(df.loc[:, include_columns])

    # Reset display options
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')


# ..............................................................................
def run(data, show=False, plot=False):

    df = _add_stats(pd.DataFrame.from_dict(_run_backtest(data)))

    df_pos = df.copy()

    if show:
        _show(df)

    positions = position.Positions(df_pos, symbol)

    # for pos in positions:
    #     print(pos)
    #     for trade in pos.trades:
    #         print(trade)

    print(f"Total number of trades: {len(positions)}")
    print(f"Profit Factor: {positions.profit_factor:.2f}")

    strategy_stats = st.calculate_statistics(df['b.value'].copy().to_numpy())
    strategy_stats = {k: round(v, 3) for k, v in strategy_stats.items()}

    hodl_stats = st.calculate_statistics(df['hodl.value'].copy().to_numpy())
    hodl_stats = {k: round(v, 3) for k, v in hodl_stats.items()}

    logger.info(strategy_stats)
    logger.info(hodl_stats)

    if plot:
        plot_title = f'{strategy.name} - {symbol} ({interval})'
        Chart(df=df, title=plot_title, style='backtest').show()


def test_find_positions(data: dict):
    fp.find_positions_with_dict(data)

    assert "position" in data, "'position' not found in data dictionary"


# ==================================================================------========== #
#                                       MAIN                                         #
# ================================================================================== #
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


    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            bt.run(strategy, data_pre[i], initial_capital, risk_level)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    et = time.perf_counter()
    print(f'length data: {len(data_pre[0]["close"])} periods')
    print(f"average execution time: {((et - st)*1_000_000/runs):.2f} microseconds")