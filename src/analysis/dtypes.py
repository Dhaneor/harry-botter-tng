#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 06 01:28:53 2024

@author: dhaneor
"""
import numpy as np


OHCLV_DTYPE = np.dtype([
    ('timestamp', np.int64),  # timestamp of the OHLCV data
    ('open', np.float32),  # price at the open
    ('high', np.float32),  # price at the highest
    ('low', np.float32),  # price at the lowest
    ('close', np.float32),  # price at the close
    ('volume', np.float64),  # volume of the trade
])


SIGNALS_DTYPE = np.dtype([
    ('open_long', np.bool_),
    ('close_long', np.bool_),
    ('open_short', np.bool_),
    ('close_short', np.bool_),
    ('combined', np.float64),  # one column representation of all signals
])


POSITION_DTYPE = np.dtype([
    ('position', np.int8),  # position of the instrument (0=none, 1=long,  -1=short)
    ('qty', np.float64),  # quantity of the instrument (can be negative for shorts)
    ('quote_qty', np.float64),  # current quote asset balance for the asset
    ('entry_price', np.float64),  # entry price for the position
    ('duration', np.uint16),  # duration of the position (trading periods)
    ('equity', np.float64),  # current equity/value of the position
    ('buy_qty', np.float64),  # quantity bought at the current step
    ('buy_price', np.float64),  # price for buys (open price / stop price)
    ('sell_qty', np.float64),  # quantity sold at the current step
    ('sell_price', np.float64),  # price for sells (open price / stop price)
    ('fee', np.float64),  # fee(s) for the trade(s) at this step (in quote asset)
    ('slippage', np.float64),  # slippage for the trade(s) at this step (in quote asset)
    ('asset_weight', np.float32),  # weight for the asset in the portfolio
    ('strategy_weight', np.float32),  # weight of the strategy in the portfolio
])


PORTFOLIO_DTYPE = np.dtype([
    ('quote_balance', np.float64),  # Balance of the quote asset
    ('equity', np.float64),  # sum of position values
    ('total_value', np.float64),  # equity + quote balance
    ('leverage', np.float64),  # Current leverage
])
