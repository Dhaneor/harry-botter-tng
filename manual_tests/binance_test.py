#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:11:33 2021

@author dhaneor
"""

import sys
import os
import logging

LOG_LEVEL = "INFO"
logger = logging.getLogger("main")
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)


# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from src.exchange.binance_ import Binance  # noqa: E402, F401


def test_get_ohlcv(symbol: str, interval: str, start: int, end: int):
    binance = Binance()
    ohlcv = binance.get_ohlcv(symbol=symbol, interval=interval, start=start, end=end)
    if ohlcv:
        print(f"Ohlcv for {symbol} at {interval} interval:")
        print(ohlcv)
    else:
        print("No Ohlcv data found")


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15m"
    start = "one year ago UTC"
    end = "now UTC"

    test_get_ohlcv(symbol="BTCUSDT", interval="15m", start=start, end=end)