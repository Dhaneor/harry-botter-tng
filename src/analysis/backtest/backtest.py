#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 13 02:00:23 20235
@author dhaneor
"""

import numpy as np

from numba import float64, boolean
from numba.experimental import jitclass
from numba import types
from numba import from_dtype
from typing import Callable

from analysis import MarketDataStore
from analysis.dtypes import SIGNALS_DTYPE, POSITION_DTYPE, PORTFOLIO_DTYPE

WARMUP_PERIODS = 200

# Define the configuration class specification
config_spec = [
    ("initial_capital", float64),
    ("rebalance_position", boolean),
    ("increase_allowed", boolean),
    ("decrease_allowed", boolean),
    ("minimum_change", float64),
    ("fee_rate", float64),
    ("slippage_rate", float64),
]


@jitclass(config_spec)
class Config:
    def __init__(
        self, 
        initial_capital, 
        rebalance_position = True, 
        increase_allowed = True, 
        decrease_allowed = True,
        minimum_change = 0.1,
        fee_rate = 0.001,
        slippage_rate = 0.001, 
    ):
        self.initial_capital = initial_capital
        self.rebalance_position = rebalance_position
        self.increase_allowed = increase_allowed
        self.decrease_allowed = decrease_allowed
        self.minimum_change = minimum_change
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate


# ================================ BackTest class definition ===========================
# Define the LeverageArray2D specification
LeverageArray2D = types.Array(types.float32, 2, "C")
# Define the SignalRecord specification
#
# NOTE: The signals array is a 3D array with dimensions (time, 
#       instruments, signals), and it has a custom dtype (structured 
#       array) for the signal records. Numba requires to define this 
#       in the following way:
#       • convert the dtype into a Numba Record type (with a special 
#         helper function)
#       • define the array with the Record type and the dimensions
SignalRecord = from_dtype(SIGNALS_DTYPE)
SignalArray3D = types.Array(SignalRecord, 3, "C")

# ... the PositionRecord specification (same procedure as for the signals)
PositionRecord = from_dtype(POSITION_DTYPE)
PositionArray3D = types.Array(PositionRecord, 3, "C")
# ... and the PortfolioRecord specification
PortfolioRecord = from_dtype(PORTFOLIO_DTYPE)
PortfolioArray2D = types.Array(PortfolioRecord, 2, "C")

spec = [
    ("market_data", MarketDataStore.class_type.instance_type),
    ("leverage", LeverageArray2D),
    ("signals", SignalArray3D),
    ("config", Config.class_type.instance_type),
    ("positions", PositionArray3D),
    ("portfolio", PortfolioArray2D),
] 


# ================================ BackTest class definition ===========================
@jitclass(spec)
class BackTest:
    def __init__(
        self,
        market_data: MarketDataStore,
        leverage: np.ndarray,
        signals: np.ndarray,
        config: Config,
    ):
        # assert market_data.shape[0] == leverage.shape[0] == signals.shape[0],\
        #       "Market data/leverage/signals arrays must have the same length."
        
        # assert market_data.shape[0] > WARMUP_PERIODS, \
        #     f"Market data must have at least {WARMUP_PERIODS + 1} periods."

        self.market_data = market_data
        self.leverage = leverage
        self.signals = signals
        self.config = config

        self.positions = np.zeros(self.signals.shape, dtype=POSITION_DTYPE)
        
        # initialize Portfolio array
        self.portfolio = np.zeros(
            shape=(signals.shape[0], signals.shape[2]), 
            dtype=PORTFOLIO_DTYPE
            )  

    # def _initialize_positions(self) -> np.ndarray:
    #     """Initialize the portfolios array with the structured dtype.

    #     Will return a 3D array with dimensions (time, instruments, strategies)
    #     and a custom dtype for the portfolio records. The cutstom dtype is
    #     defined in a separate file to allow access for other modules.
        
    #     .. code-block:: python
    #     POSITION_DTYPE = np.dtype([
    #         ('position', np.int8),  # 0 = none, 1 = long,  -1 = short
    #         ('qty', np.float64),  # quantity (can be negative for shorts)
    #         ('entry_price', np.float64),  # entry price for the position
    #         ('duration', np.int32),  # duration (trading periods)
    #         ('equity', np.float64),  # current equity/value of the position
    #         ('change_qty', np.float64),  # change (= buy/sell qty)
    #         ('change_price', np.float64),  # buy/sell price (open price / stop price)
    #         ('asset_weight', np.float32),  # weight for the asset in the portfolio
    #         ('strat_weight', np.float32),  # weight of the strategy in the portfolio
    #     ])
    #     """
    #     return np.zeros(self.signals.shape, dtype=POSITION_DTYPE)

    def run(self):
        periods, markets, strategies = self.signals.shape

        for p in range(WARMUP_PERIODS, periods):
            self.positions[p] = self.positions[p - 1]  # copy previous portfolio
            
            for s in range(strategies):
                for m in range(markets):
                    self._process_single(p, m, s)
                self._update_portfolio_equity(p, s)
        
        return self.positions
    
    def _update_portfolio_equity(self, p: int, s: int):
        # Sum the equity values for all markets for the current period and strategy
        asset_value= np.sum(self.positions[p, :, s]['equity'])
        quote_balance = self.portfolio[p, s]['quote_balance']
        
        total_value = asset_value + quote_balance
        leverage = abs(asset_value) / total_value if total_value > 0 else 0
         
        self.portfolio[p, s]['leverage'] = leverage    
        self.portfolio[p, s]['equity'] = asset_value
        self.portfolio[p, s]['total_value'] = total_value

    def _process_single(self, p: int, m: int, s: int):

        portfolio = self.positions[p, m, s]
        position = portfolio["position"]    
        signal = self.signals[p-1, m, s]
        open_long = signal["open_long"] == 1
        close_long = signal["close_long"] == 1
        open_short = signal["open_short"] == 1
        close_short = signal["close_short"] == 1

        self.positions[p, m, s]["change_qty"] = 0
        self.positions[p, m, s]["change_price"] = np.nan

        if position == 1:
            if close_long or open_short:
                price = self.market_data.close[p, m]
                self._close_long_position(p, m, s, price)
            else:
                self._update_long_position(p, m, s)

        if position == -1:
            if close_short or open_long:
                price = self.market_data.close[p, m]
                self._close_long_position(p, m, s, price)
            else:
                self._update_short_position(p, m, s)

        if position == 0:
            if open_long:
                self._open_long_position(p, m, s)
            elif open_short:
                self._open_short_position(p, m, s)
        
        self.positions[p, m, s]["equity"] = \
            self.market_data.close[p, m] * self.positions[p, m, s]["qty"]
        # self.portfolio[p, s] = self.positions[p, m, s]["equity"]

    def _open_long_position(self, p, m, s):
        exposure_quote = self._get_target_exposure(p, m, s)
        price = self.market_data.open_[p, m]
        # calculate the fee & slippage
        fee , slippage = self._calculate_fee_and_slippage(exposure_quote, price)
        
        # calculate the change in quantity (buy/sell)
        base_qty = (exposure_quote - fee - slippage) / price

        self.positions[p, m, s]["position"] = 1
        self.positions[p, m, s]["qty"] += base_qty
        self.positions[p, m, s]["entry_price"] = price
        self.positions[p, m, s]["duration"] += 1
        self.positions[p, m, s]["change_qty"] = base_qty
        self.positions[p, m, s]["change_price"] = price        
        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage

        return
    
    def _update_long_position(self, p, m, s):
        # assign required values to local variables for convenience
        portfolio = self.positions[p, m, s]
        
        # determine buy/sell price (open price / stop price)
        # this is the open price for now (unitl stop orders are implemented)
        price = self.market_data.open_[p, m]

        # calculate the change in exposure
        current_exposure = portfolio["equity"]
        target_exposure = self._get_target_exposure(p, m, s)
        change_exposure = target_exposure - current_exposure

        change_pct = abs(change_exposure / current_exposure) if current_exposure > 0 else 1

        if change_pct < self.config.minimum_change:
            return

        if change_exposure > 0 and not self.config.increase_allowed:
            return

        if change_exposure < 0 and self.config.decrease_allowed:
            return
        
        # calculate the fee & slippage
        fee , slippage = self._calculate_fee_and_slippage(change_exposure, price)

        # calculate the change in quantity (buy/sell)
        change_qty = (change_exposure - fee - slippage) / price

        self.positions[p, m, s]["qty"] += change_qty
        self.positions[p, m, s]["duration"] += 1
        self.positions[p, m, s]["change_qty"] += change_qty
        self.positions[p, m, s]["change_price"] = price
        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage

    def _close_long_position(self, p, m, s, price):
        portfolio = self.positions[p, m, s]
        
        close_exposure = portfolio["qty"] * price
        fee, slippage = self._calculate_fee_and_slippage(close_exposure, price)

        self.portfolio[p, s]["quote_balance"] += close_exposure - fee - slippage
        
        # Update portfolio
        self.positions[p, m, s]["qty"] = 0
        self.positions[p, m, s]["position"] = 0
        self.positions[p, m, s]["duration"] = 0
        self.positions[p, m, s]["change_qty"] -= portfolio["qty"]
        self.positions[p, m, s]["change_price"] = price
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage
        self.positions[p, m, s]["equity"] = 0

    def _open_short_position(self, p, m, s):
        exposure_quote = self._get_target_exposure(p, m, s)
        price = self.market_data.open_[p, m]
        # calculate the fee & slippage
        fee , slippage = self._calculate_fee_and_slippage(exposure_quote, price)
        
        # calculate the change in quantity (buy/sell)
        base_qty = (exposure_quote - fee - slippage) / price

        self.positions[p, m, s]["position"] = -1
        self.positions[p, m, s]["qty"] = -base_qty
        self.positions[p, m, s]["entry_price"] = price
        self.positions[p, m, s]["duration"] += 1
        self.positions[p, m, s]["change_qty"] = -base_qty
        self.positions[p, m, s]["change_price"] = price        
        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage

        return

    def _update_short_position(self, p, m, s):
        # assign required values to local variables for convenience
        portfolio = self.positions[p, m, s]
        
        # determine buy/sell price (open price / stop price)
        # this is the open price for now (unitl stop orders are implemented)
        price = self.market_data.open_[p, m]

        # calculate the change in exposure
        current_exposure = portfolio["equity"]
        target_exposure = self._get_target_exposure(p, m, s)
        change_exposure = target_exposure - current_exposure

        change_pct = abs(change_exposure / current_exposure) if current_exposure > 0 else 1

        if change_pct < self.config.minimum_change:
            return

        if change_exposure > 0 and not self.config.increase_allowed:
            return

        if change_exposure < 0 and self.config.decrease_allowed:
            return
        
        # calculate the fee & slippage
        fee , slippage = self._calculate_fee_and_slippage(change_exposure, price)

        # calculate the change in quantity (buy/sell)
        change_qty = -1 * (change_exposure - fee - slippage) / price

        self.positions[p, m, s]["qty"] += change_qty
        self.positions[p, m, s]["duration"] += 1
        self.positions[p, m, s]["change_qty"] = change_qty
        self.positions[p, m, s]["change_price"] = price
        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage
    
    def _close_short_position(self, p, m, s, price):
        portfolio = self.positions[p, m, s]
        
        close_exposure = portfolio["qty"] * price
        fee, slippage = self._calculate_fee_and_slippage(close_exposure, price)
        
        # Update portfolio
        self.positions[p, m, s]["qty"] = 0
        self.positions[p, m, s]["position"] = 0
        self.positions[p, m, s]["duration"] = 0
        self.positions[p, m, s]["change_qty"] = -portfolio["qty"]
        self.positions[p, m, s]["change_price"] = price
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage
        self.positions[p, m, s]["equity"] = 0

    def _get_target_exposure(self, p, m, s):
        asset_weight = self.positions[p, m, s]["asset_weight"]
        strategy_weight = self.positions[p, m, s]["strategy_weight"]
        exposure_quote = self.portfolio[p, s]["equity"] \
            * asset_weight \
            * strategy_weight \
            * self.leverage[p, m]
        return exposure_quote
    
    def _calculate_fee_and_slippage(self, change_quote, price):
        # calculate fee & slippage
        fee = self.config.fee_rate * change_quote
        slippage = self.config.slippage_rate * change_quote
        return fee, slippage


# ======================================================================================
def _process_one_parameter_combination(
    open_prices: np.ndarray,
    closeprices: np.ndarray,
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
            long_exit = signals[p - 1, s]["closelong"]
            short_entry = signals[p - 1, s]["open_short"]
            short_exit = signals[p - 1, s]["closeshort"]
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
    POSITION_DTYPE = np.dtype(
        [
            ("position", np.int8),
            ("balance_base", np.float64),
            ("balance_quote", np.float64),
            ("equity", np.float64),
            ("drawdown", np.float64),
            ("max_drawdown", np.float64),
        ]
    )

    portfolios = np.zeros_like(signals, dtype=POSITION_DTYPE)

    param_combinations = signals.shape[2]

    for c in range(param_combinations):
        _process_one_parameter_combination(
            open_prices=market_data.open_,
            closeprices=market_data.close,
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
