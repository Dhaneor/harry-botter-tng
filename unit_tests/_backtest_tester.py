#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import logging
from typing import Union

LOG_LEVEL = "DEBUG"
LOGGER = logging.getLogger('main')
LOGGER.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
)
ch.setFormatter(formatter)

LOGGER.addHandler(ch)

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------

# from src.backtest.backtest_vector import Backtest
from src.analysis.backtest.backtest import Backtest  # noqa: E402, F401
from src.analysis.backtest.result_stats import calculate_stats  # noqa: E402, F401
from src.models.symbol import Symbol  # noqa: E402, F401
from src.plotting.minerva import BacktestChart  # noqa: E402, F401
from util.timeops import execution_time  # noqa: E402, F401

bt = Backtest(exchange='kucoin')


# =============================================================================
@execution_time
def test_run(symbol: str, interval: str, strategy: str,
             start: Union[str, int], end: Union[str, int],
             leverage: float, stop_loss_strategy, risk_level: int,
             draw_chart: bool = False):

    initial_capital = 0.1 if 'BTC' in symbol[-3:] else 1000
    df = None

    strategy_params = {
        'long_allowed': True,
        'short_allowed': False,
        'initial_capital': initial_capital,
        'mode': 'backtest'
    }

    sl_params = {
        'type': 'trailing',
        'mode': 'atr',
        'atr factor': 3,
        'percent': 30
    }

    bt = Backtest(exchange='kucoin')

    for _ in range(2):
        df = bt.run(
            symbol=symbol,
            interval=interval,
            strategy=strategy,
            start=start,
            end=end,
            leverage=leverage,
            risk_level=risk_level,
            strategy_params=strategy_params,
            stop_loss_strategy=stop_loss_strategy,
            stop_loss_params=sl_params,
        )

    if df is not None:
        df = calculate_stats(df, initial_capital)
        bt.show_overview(df)

        if draw_chart:
            df.loc[~(df['position'] == 0), 'p.actv'] = True
            df.rename(
                columns={'buy_size': 'buy.amount', 'sell_size': 'sell.amount'},
                inplace=True
            )

            chart = BacktestChart(
                df=df,
                title=f'{symbol} ({interval})',
                color_scheme='day'
            )
            chart.draw()
    else:
        print('ain´t got no dataframe to draw chart from! :(')


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    start = -2000  # 'August 18, 2018 00:00:00'
    end = 'July 22, 2023 00:00:00'

    # for _ in range(2):
    test_run(
        symbol='ETH-BTC',
        interval='1d',
        strategy='Pure Keltner',
        start=start,
        end=end,
        leverage=4,
        risk_level=4,
        stop_loss_strategy=None,
        draw_chart=False
    )

    # test_vector(
    #     symbol_name= 'BTC-USDT',
    #     interval='1d',
    #     strategy='Pure Moving Average Cross',
    #     start=start,
    #     end=end,
    #     leverage=4,
    #     risk_level=3,
    #     stop_loss_strategy='atr',
    #     verbose=False,
    #     draw_chart=False
    # )