#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import logging
import random, string
from pprint import pprint 

# ----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.broker.ganesh import Ganesh
from src.helpers.timeops import execution_time
from config import CREDENTIALS

# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

GANESH = Ganesh(
    exchange='kucoin', market='CROSS MARGIN', credentials=CREDENTIALS
)

# =============================================================================
@execution_time
def test_get_server_time():
    print(GANESH.server_time)

@execution_time
def test_get_system_status():
    print(GANESH.system_status)

@execution_time
def test_get_ticker():
    pprint(GANESH.tickers[0])

@execution_time    
def test_get_currencies():
    pprint(GANESH.currencies[0])
    
@execution_time
def test_get_symbol():
    pprint(GANESH.get_symbol('BTC-USDT'))

@execution_time    
def test_get_all_symbols():
    pprint(random.choice(GANESH.get_all_symbols()))
    
@execution_time    
def test_get_symbols():
    # print([s['symbol'] for s in GANESH.symbols])
    print(len(GANESH.symbols))
    
def test_valid_assets():
    print(list(sorted(GANESH.valid_assets)))
    
# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    # test_get_server_time()
    # test_get_system_status()
    
    # test_get_ticker()
    # test_get_currencies()
    
    # test_get_symbol()
    test_valid_assets()
    
    # for _ in range(3):
    #     test_get_all_symbols()

    # for _ in range(3):
    #     test_get_symbols()