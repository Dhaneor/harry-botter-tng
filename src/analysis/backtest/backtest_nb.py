#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides backtest components.

Created on Jan 13 02:00:23 2025
@author dhaneor
"""

import logging
import numpy as np

from numba import int16, float64, boolean, types, from_dtype
from numba.experimental import jitclass
from typing import Tuple

from analysis.models.market_data import MarketDataStoreJIT as MarketDataStore
from analysis.dtypes import POSITION_DTYPE, PORTFOLIO_DTYPE

logger = logging.getLogger("main.backtest")

WARMUP_PERIODS = 200  # not included for portfolio calculations, needed for signals

# ================================= Config class definition ============================
config_spec = [
    ("initial_capital", float64),
    ("max_leverage", float64),
    ("rebalance_freq", int16),
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
        max_leverage = 1.0,
        rebalance_freq = 0,  # in trading periods
        rebalance_position = True, 
        increase_allowed = True, 
        decrease_allowed = True,
        minimum_change = 0.1,
        fee_rate = 0.001,
        slippage_rate = 0.001, 
    ):
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq
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
#       array) for the signal records (not anymore, but the procedure
#       still aplplies to the arrays for Position and Portfolio). 
#       Numba requires to define this in the following way:
#       • convert the dtype into a Numba Record type (with a special 
#         helper function)
#       • define the array with the Record type and the dimensions
SignalRecord = from_dtype(np.float64)
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


# ================================== BackTest class ====================================
@jitclass(spec)
class BackTestCore:
    def __init__(
        self,
        market_data: MarketDataStore,
        leverage: np.ndarray,
        signals: np.ndarray,
        config: Config,
    ):
        self.market_data = market_data
        self.leverage = leverage
        self.signals = signals
        self.config = config

        # initialize Position array
        self.positions = np.zeros(self.signals.shape, dtype=POSITION_DTYPE)

        _, markets, strategies = self.signals.shape
        for s in range(strategies):
            for m in range(markets):
                a_weight = 1 / markets
                s_weight = 1 / strategies
                initial_capital = self.config.initial_capital * a_weight * s_weight

                self.positions[WARMUP_PERIODS - 1, m, s]["asset_weight"] = a_weight
                self.positions[WARMUP_PERIODS - 1, m, s]["strategy_weight"] = s_weight
                self.positions[WARMUP_PERIODS - 1, m, s]["quote_qty"] = initial_capital
                self.positions[WARMUP_PERIODS - 1, m, s]["equity"] = initial_capital
        
        # initialize Portfolio array
        self.portfolio = np.zeros(
            shape=(signals.shape[0], signals.shape[2]), 
            dtype=PORTFOLIO_DTYPE
            )
        
        capital_per_strategy = self.config.initial_capital / strategies
        
        for s in range(strategies):
            self.portfolio[WARMUP_PERIODS - 1, s]["quote_balance"] = capital_per_strategy
            self._update_portfolio(WARMUP_PERIODS - 1, s)

    def run(self):
        periods, markets, strategies = self.signals.shape
        
        # process the signals for each trading interval (period)
        for p in range(WARMUP_PERIODS, periods):
            # copy all from previous period to initialize the current period
            self.positions[p] = self.positions[p - 1] 
            self.portfolio[p] = self.portfolio[p - 1]

            # ... strategy by strategy
            for s in range(strategies):
                # market by market
                for m in range(markets):
                    self._process_period(p, m, s)
                
                self._update_portfolio(p, s)
        
        return self.positions, self.portfolio
    
    # ..................................................................................
    def _update_portfolio(self, p: int, s: int):
        # Sum the equity values for all markets for the current period and strategy
        asset_value= np.sum(self.positions[p, :, s]['equity'])
        quote_balance = self.portfolio[p, s]['quote_balance']
        
        total_value = asset_value + quote_balance
        leverage = abs(asset_value) / total_value if total_value > 0 else 0
         
        self.portfolio[p, s]['leverage'] = leverage    
        self.portfolio[p, s]['equity'] = asset_value
        self.portfolio[p, s]['total_value'] = total_value

    def _process_period(self, p: int, m: int, s: int):
        self.positions[p, m, s]["buy_price"] = np.nan
        self.positions[p, m, s]["sell_price"] = np.nan
        self.positions[p, m, s]["buy_qty"] = np.nan
        self.positions[p, m, s]["sell_qty"] = np.nan
        self.positions[p, m, s]["fee"] = 0.0
        self.positions[p, m, s]["slippage"] = 0.0

        record = self.positions[p, m, s]
        position = record["position"]    
        
        signal = self.signals[p-1, m, s]
        prev_signal = self.signals[p-2, m, s] if p > 1 else 0
        
        open_long = signal > 0 and prev_signal <= 0
        close_long = signal <= 0 and prev_signal > 0
        open_short = signal < 0 and prev_signal >= 0
        close_short = signal >= 0 and prev_signal < 0

        if position == 1:
            if close_long or open_short:
                self._close_position(p, m, s)
            else:
                self._update_position(p, m, s)

        if position == -1:
            if close_short or open_long:
                self._close_position(p, m, s)
            else:
                self._update_position(p, m, s)

        if position == 0:
            if open_long:
                self._open_position(p, m, s, 1)
            elif open_short:
                self._open_position(p, m, s, -1)
        
        self.positions[p, m, s]["equity"] = \
            self.market_data.close[p, m] \
            * self.positions[p, m, s]["qty"] \
            + self.positions[p, m, s]["quote_qty"] 

    # ............................ PROCESSINNG POSITIONS ...............................
    def _open_position(self, p: int, m: int, s: int, type: int):
        price = self.market_data.open_[p, m]
        exposure_quote, _ = self._calculate_change_exposure(p, m, s, price)

        self.positions[p, m, s]["position"] = type
        self.positions[p, m, s]["entry_price"] = price
        self.positions[p, m, s]["duration"] = 1

        if exposure_quote > 0:
            self._process_buy(p, m, s, exposure_quote, price)
        elif exposure_quote < 0:
            self._process_sell(p, m, s, -exposure_quote, price)

    def _close_position(self, p, m, s):
        price = self.market_data.close[p, m]
        portfolio = self.positions[p, m, s]
        position = self.positions[p, m, s]["position"]

        if position == 0:
            return
        
        change_quote = portfolio["qty"] * price
        fee, slippage = self._calculate_fee_and_slippage(change_quote)

        self.positions[p, m, s]["quote_qty"] += change_quote - fee - slippage
        
        # Update portfolio
        if position == 1:
            self.positions[p, m, s]["sell_qty"] = portfolio["qty"]
            self.positions[p, m, s]["sell_price"] = price
        elif position == -1:
            self.positions[p, m, s]["buy_qty"] = abs(portfolio["qty"])
            self.positions[p, m, s]["buy_price"] = price

        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage
        self.positions[p, m, s]["equity"] = 0

        self.positions[p, m, s]["qty"] = 0
        self.positions[p, m, s]["position"] = 0
        self.positions[p, m, s]["duration"] = 0
        self.positions[p, m, s]["entry_price"] = 0

    def _update_position(self, p, m, s):
        position_type = self.positions[p, m, s]["position"]
        price = self.market_data.open_[p, m]
        change_exposure, change_pct = self._calculate_change_exposure(p, m, s, price)

        self.positions[p, m, s]["duration"] += 1

        if not self.config.rebalance_position \
        or change_pct < self.config.minimum_change:
            return
        
        fee , slippage = self._calculate_fee_and_slippage(change_exposure)
        change_qty = (change_exposure - fee - slippage) / price

        if position_type == 1:
            if change_qty > 0 and not self.config.increase_allowed:
                return
            elif change_qty < 0 and not self.config.decrease_allowed:
                return

        if position_type == -1:
            if change_qty < 0 and not self.config.increase_allowed:
                return
            elif change_qty > 0 and not self.config.decrease_allowed:
                return

        if change_qty > 0:
            self._process_buy(p, m, s, change_exposure, price)
        elif change_qty < 0:
            self._process_sell(p, m, s, -change_exposure, price)
    
    # ..................................................................................
    def _process_buy(self, p, m, s, quote_qty, price):
        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = (quote_qty - fee - slippage) / price

        self.positions[p, m, s]["buy_qty"] = base_qty
        self.positions[p, m, s]["buy_price"] = price
        self.positions[p, m, s]["quote_qty"] -= quote_qty

        self.positions[p, m, s]["qty"] += base_qty
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage

    def _process_sell(self, p, m, s, quote_qty, price):
        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = quote_qty / price

        self.positions[p, m, s]["sell_qty"] = abs(base_qty)
        self.positions[p, m, s]["sell_price"] = price
        self.positions[p, m, s]["quote_qty"] += quote_qty - fee - slippage

        self.positions[p, m, s]["qty"] -= base_qty
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage

    def _calculate_change_exposure(self, p, m, s, price) -> Tuple[float, float]:
        """Calculate the change in exposure for a given position/asset.
        
        Parameters:
        -----------
        p (int): period index
        m (int): market index
        s (int): strategy index

        Returns:
        --------
        tuple (float, float): change in exposure (quote asset) and percentage change
        """
        
        # calculate the current exposure
        current_exposure = self.positions[p - 1, m, s]["qty"] * price

        # calulate the target exposure
        equity = current_exposure + self.positions[p - 1, m, s]["quote_qty"]
        target_exposure =  equity * self.signals[p-1, m, s] * self.leverage[p, m]

        # even if the leverage value is smaller that the max allowed
        # leverage, signals can be >1 or <-1, so we need to check the 
        # effective leverage again and adjust if necessary
        effective_leverage = np.abs(target_exposure / equity)

        if effective_leverage > self.config.max_leverage:
            target_exposure /= (effective_leverage / self.config.max_leverage)

        # calculate the change (absolute & percentage)      
        change = target_exposure - current_exposure

        return (
            change, 
            change / current_exposure if current_exposure > 0 else effective_leverage,
        )

    def _calculate_leverage(self, p, m, s) -> float:
        price = self.market_data.open_[p, m]
        current_exposure = self.positions[p - 1, m, s]["qty"] * price
        equity = current_exposure + self.positions[p - 1, m, s]["quote_qty"]
        
        return abs(current_exposure / equity)  

    def _calculate_fee_and_slippage(self, change_quote):
        # calculate fee & slippage
        fee = self.config.fee_rate * change_quote
        slippage = self.config.slippage_rate * change_quote
        return abs(fee), abs(slippage)


# ======================================================================================
def run_backtest(
    market_data: MarketDataStore, 
    leverage: np.ndarray, 
    signals: np.ndarray, 
    config: Config
) -> np.ndarray:

    _, assets, strategies = signals.shape    
    logger.info(
        "shape of signals array: %s (%s backtests)", signals.shape, assets * strategies
    )

    bt = BackTestCore(market_data, leverage, signals, config)

    return bt.run()
