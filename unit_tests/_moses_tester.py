#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work
import sys
import os
import time

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from staff.moses import Moses
from staff.hermes import Hermes
from plotting.minerva02 import MosesChart 

# =============================================================================
def test_moses_analyze(symbol:str, interval:str, start:str, end:str):
    
    hermes = Hermes(exchange='binance')

    res = hermes.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
        )
    
    if res.get('success'):
        df = res.get('message')

    moses = Moses()
    moses.set_sl_strategy('moving average', 
                          {'atr factor' : 1,
                           'percent' : 10}
                          )
    df = moses.get_stop_loss_prices(df=df)
    
    print(df)
    moses.draw_chart(df)
    

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    test_moses_analyze(symbol='XRPUSDT', 
                        interval='1d', 
                        start = 'January 01, 2021 00:00:00',
                        end='July 01, 2022 00:00:00')
