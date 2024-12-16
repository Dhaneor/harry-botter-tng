#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 16:49:20 2022

@author dhaneor
"""
import sys
import os
import logging
from pprint import pprint

from vectorbtpro import Data


LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.analysis.models.vbt_data import LiveData
from util.timeops import execution_time

exchange = 'kucoin'
symbols = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'BNB/USDT', 'XDC/USDT',
    'QNT/USDT', 'XLM/USDT', 'KCS/USDT'
]
# symbols = ['BTC/USDT']

# -----------------------------------------------------------------------------
@execution_time
def test_fetch_symbol_multiple():
    res = LiveData.fetch_symbol_multiple(
        exchange=exchange, symbols=symbols, interval='1 minute'
    )

    data = LiveData.from_data(data=res, silence_warnings=True) #type:ignore

    print(data.stats())



# =============================================================================
if __name__ == '__main__':
    test_fetch_symbol_multiple()