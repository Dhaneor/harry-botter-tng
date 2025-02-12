#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 01:28:23 2025

@author dhaneor
"""

from analysis.models.portfolio import Buy, Sell, Position

# Create a test action
buy = Buy(1625097600000, 35000.0, 35000.0)

print(f"Action type: {buy.type}")
print(f"Action quantity: {buy.qty}")
print(f"Action price: {buy.price}")
print(buy)

sell = Sell(1625098200000, 0.5, 38000.0)

# Create a test position
position = Position("BTCUSDT")
position.add_action(buy)
position.add_action(sell)

print(f"Position symbol: {position.symbol}")
print(f"Position actions: {position.get_actions()}")
print(f"Position current quantity: {position.current_qty}")