#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a JIT class for holding position information.

Created on Jan 12 00:45:23 2025

@author dhaneor
"""
import numpy as np
from numba import float64, int64, uint8
from numba.experimental import jitclass
from numba.types import string

spec = [
    ("timestamp", int64),
    ("symbol", string(10)),  # Assuming max length of 10 for symbol
    ("action", string(5)),   # Assuming max length of 5 for action (e.g., "BUY", "SELL")
    ("amount", float64),
    ("fee", float64),
]

@jitclass(spec)
class Action:
    """A class to store information about a trade action."""

    def __init__(
        self, 
        timestamp: np.int64, 
        symbol: str,
        action: str,
        amount: np.float64, 
        fee: np.float64
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.action = action
        self.amount = amount
        self.fee = fee




# spec = [
#     ("timestamp", int64[:, :]),
#     ("open_", float64[:, :]),
#     ("high", float64[:, :]),
#     ("low", float64[:, :]),
#     ("close", float64[:, :]),
#     ("volume", float64[:, :]),
#     ("log_returns", float64[:, :]),
#     ("atr", float64[:, :]),
#     ("annual_vol", float64[:, :]),
# ]