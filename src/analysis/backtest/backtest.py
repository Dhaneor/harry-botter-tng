#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 13 02:00:23 20235
@author dhaneor
"""
import numpy as np
from numba import njit
from typing import Callable

from analysis import MarketDataStore


def _process_one_parameter_combination(
        open_prices: np.ndarray,
        close_prices: np.ndarray,
        leverage: np.ndarray,  # 2D - shape (periods, symbols)
        signals: np.ndarray,  # 2D - shape (periods, symbols)
        portfolio: np.ndarray,  # 2D - shape (periods, symbols)
        config
):

    periods =portfolio.shape[0]
    symbols = portfolio.shape[1]

    for p in range(1, periods):
        for s in range(symbols):
            long_entry = signals[p-1, s]["open_long"]
            long_exit = signals[p-1, s]["close_long"]
            short_entry = signals[p-1, s]["open_short"]
            short_exit = signals[p-1, s]["close_short"]
            active_position = portfolio['position']

            if active_position != 1 and long_entry:
                portfolio['position'][p, s] = 1
                # Buy at current price
                # Calculate return
                # Update portfolio
                ...

    return 


def run_backtest_nb(
        market_data: MarketDataStore,  # 4x 2D - shape (periods, symbols)
        leverage: np.ndarray,  # 2D - shape (periods, symbols)
        signals: np.ndarray,  # 3D - shape (periods, symbols, param_combinations)
        portfolios: np.ndarray,  # 3D - shape (periods, symbols, param_combinations)
        config
):
    portfolio_dtype = np.dtype([
        ('position', np.int8),
        ('balance_base', np.float64),
        ('balance_quote', np.float64),
        ('equity', np.float64),
        ('drawdown', np.float64),
        ('max_drawdown', np.float64),
    ])

    portfolios = np.zeros_like(signals, dtype=portfolio_dtype)

    param_combinations = signals.shape[2]

    for c in range(param_combinations):  
        _process_one_parameter_combination(
            open_prices=market_data.open_,
            close_prices=market_data.close_,
            leverage=leverage, 
            signals=signals[:, :, c], 
            portfolio=portfolios[:, :, c], 
            config=config
        )


def backtest(
    market_data: MarketDataStore,
    leverage: np.ndarray,
    signals: np.ndarray,
    config,
    rebalance_fn: Callable = None,
):

    run_backtest_nb(
        market_data=market_data, 
        leverage=leverage, 
        signals=signals,
        config=config
        )