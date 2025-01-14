#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 06 21:12:20 2023

@author dhaneor
"""
from numba import types
from numba.experimental import jitclass

spec = [
    ("symbol", types.unicode_type),
]

@jitclass(spec)
class Position:
    def __init__(self, symbol: str):
        self.symbol = symbol


if __name__ == "__main__":
    position = Position("AAPL")
    print(position.symbol)  # Output: AAPL