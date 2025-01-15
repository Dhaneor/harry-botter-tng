#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 13 02:00:23 20235
@author dhaneor
"""

import numpy as np

from numba import float64, float32, boolean
from numba.experimental import jitclass
from numba import types
from numba import from_dtype
from typing import Callable

from analysis import MarketDataStore
from analysis.dtypes import SIGNALS_DTYPE


# Define the configuration class specification
config_spec = [
    ("rebalance_position",boolean),
    ("increase_allowed", boolean),
    ("decrease_allowed", boolean),
]

@jitclass(config_spec)
class Config:
    def __init__(self, rebalance_position, increase_allowed, decrease_allowed):
        self.rebalance_position = rebalance_position
        self.increase_allowed = increase_allowed
        self.decrease_allowed = decrease_allowed


SignalRecord = from_dtype(SIGNALS_DTYPE)
SignalArray3D = types.Array(SignalRecord, 3, 'C')
LeverageArray2D = types.Array(types.float32, 2, 'C')

spec = [
    ("market_data", MarketDataStore.class_type.instance_type),
    ("leverage", LeverageArray2D),
    ("signals",  SignalArray3D),
    ("config", Config.class_type.instance_type),
    # ("rebalance_fn", optional(types.FunctionType)),
    # ("stop_order_fn", optional(types.FunctionType)),
    ("portfolios", float64[:, :, :]),
]


@jitclass(spec)
class BackTest:
    def __init__(
        self,
        market_data: MarketDataStore,
        leverage: np.ndarray,
        signals: np.ndarray,
        config: Config
        # rebalance_fn: Callable = None,
        # stop_order_fn: Callable = None,
    ):
        self.market_data = market_data
        self.leverage = leverage
        self.signals = signals
        self.config = config
        # self.rebalance_fn = rebalance_fn
        # self.stop_order_fn = stop_order_fn

        # self.portfolios = self._initialize_portfolios()

    # def _initialize_portfolios(self):
    #     # Initialize the portfolios array with the structured dtype
    #     portfolio_dtype = np.dtype([
    #         ('position', np.int8),
    #         ('balance_base', np.float64),
    #         ('balance_quote', np.float64),
    #         ('equity', np.float64),
    #         ('drawdown', np.float64),
    #         ('max_drawdown', np.float64),
    #     ])
    #     return np.zeros(self.signals.shape, dtype=portfolio_dtype)

    # def run(self):
    #     periods, symbols, param_combinations = self.signals.shape
    #     for p in range(1, periods):
    #         for s in range(symbols):
    #             for c in range(param_combinations):
    #                 self._process_time_step(p, s, c)

    #         if self.rebalance_fn and p % self.config['rebalance_period'] == 0:
    #             self._rebalance(p)

    # def _process_time_step(self, p, s, c):
    #     # Process one time step for a specific symbol and parameter combination
    #     portfolio = self.portfolios[p, s, c]
    #     signal = self.signals[p-1, s, c]

    #     # Your logic for processing signals and updating portfolio goes here
    #     # ...

    #     if self.stop_order_fn:
    #         self.stop_order_fn(portfolio, self.market_data, p, s)

    # def _rebalance(self, p):
    #     if self.rebalance_fn:
    #         self.rebalance_fn(self.portfolios[p], self.market_data[p])


def _process_one_parameter_combination(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    leverage: np.ndarray,  # 2D - shape (periods, symbols)
    signals: np.ndarray,  # 2D - shape (periods, symbols)
    portfolio: np.ndarray,  # 2D - shape (periods, symbols)
    config,
):
    periods = portfolio.shape[0]
    symbols = portfolio.shape[1]

    for p in range(1, periods):
        for s in range(symbols):
            long_entry = signals[p - 1, s]["open_long"]
            long_exit = signals[p - 1, s]["close_long"]
            short_entry = signals[p - 1, s]["open_short"]
            short_exit = signals[p - 1, s]["close_short"]
            active_position = portfolio["position"]

            if active_position != 1 and long_entry:
                portfolio["position"][p, s] = 1
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
    config,
):
    portfolio_dtype = np.dtype(
        [
            ("position", np.int8),
            ("balance_base", np.float64),
            ("balance_quote", np.float64),
            ("equity", np.float64),
            ("drawdown", np.float64),
            ("max_drawdown", np.float64),
        ]
    )

    portfolios = np.zeros_like(signals, dtype=portfolio_dtype)

    param_combinations = signals.shape[2]

    for c in range(param_combinations):
        _process_one_parameter_combination(
            open_prices=market_data.open_,
            close_prices=market_data.close_,
            leverage=leverage,
            signals=signals[:, :, c],
            portfolio=portfolios[:, :, c],
            config=config,
        )


def backtest(
    market_data: MarketDataStore,
    leverage: np.ndarray,
    signals: np.ndarray,
    config,
    rebalance_fn: Callable = None,
):
    run_backtest_nb(
        market_data=market_data, leverage=leverage, signals=signals, config=config
    )
