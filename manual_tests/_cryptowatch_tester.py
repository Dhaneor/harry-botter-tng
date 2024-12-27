#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23 10:55:23 2022

@author dhaneor
"""

import sys
import os
import time
import logging
import datetime

from pprint import pprint
from typing import Iterable, Union

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from exchange.util.cryptowatch import CryptoWatch
from util.timeops import execution_time


LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

CW = CryptoWatch()

# =============================================================================
def test_get_exchanges():
    CW.get_exchanges()

@execution_time
def test_get_markets(symbol=None):
    res =CW.get_markets(symbol=symbol)

    [LOGGER.debug(item) for item in res]
    LOGGER.debug(f'found {len(res)} markets for symbol {symbol}')

    if symbol:
        LOGGER.debug('-'*80)
        LOGGER.debug(CW._get_exchanges_where_symbol_has_a_market(symbol))

@execution_time
def test_get_ohlcv(symbol:str, interval:str):
    print('-'*150)
    print(CW.get_ohlcv('kucoin', symbol, interval).tail(3))
    print(datetime.datetime.utcnow())





# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    # test_get_exchanges()
    # test_get_markets('ETHUSDT')
    test_get_ohlcv(symbol='QNTUSDT', interval='1m')
    # test_get_ohlcv(symbol='ETHUSDT', interval='1h')
