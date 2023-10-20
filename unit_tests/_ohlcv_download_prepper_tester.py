#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 04 15:17:53 2022

@author: dhaneor
"""

import sys
import os
from time import time
from pprint import pprint

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from exchange.kucoin_ import Kucoin
from exchange.binance_ import Binance
from helpers.timeops import execution_time

# ==============================================================================
class PrepperTester:
    def __init__(self):
        self.exchange = Binance()
    
    @execution_time    
    def prepare(self, symbol:str, interval:str, start, end):
        return self.exchange._prepare_request(symbol=symbol,
                                             interval=interval,
                                             start=start,
                                             end=end)

    
# ==============================================================================
def test_prepare_single():
    symbol = 'XRP-USDT'
    intervals = ['1h', '6h', '12h' ,'1d']
    start = '2021-01-01 00:00:00'
    end = '2021-12-31 00:00:00'    
    pt = PrepperTester()
    
    for interval in intervals:
        try:
            pprint(pt.prepare(symbol, interval, start, end))
        except Exception as e:
            print(e)
        
# ==============================================================================
if __name__ == '__main__':
    test_prepare_single()
        
    
    
    
