#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a classes for portfolio and portfolio components.

Created on Feb 04 17:45:23 2025

@author dhaneor
"""

import numpy as np


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
        self.slippage = 0