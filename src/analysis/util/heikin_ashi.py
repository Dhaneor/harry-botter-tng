#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 08 17:45:23 2021

@author_ dhaneor
"""
import sys, os
import pandas as pd
import numpy as np
import timeit
from numba import njit
from pprint import pprint
# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

# -----------------------------------------------------------------------------
from staff.hermes import Hermes
from plotting.minerva import BasicChart

# ==============================================================================
@njit(cache=True)
def build_ha(open, high, low, close):
    ha_open = np.full(close.shape[0], np.nan)
    ha_high = np.full(close.shape[0], np.nan)
    ha_low = np.full(close.shape[0], np.nan)
    ha_close = (open + high + low + close) / 4
    
    ha_open[0] = open[0]
    ha_high[0] = high[0]
    ha_low[0] = low[0]

    for i in range(ha_open.shape[0]):
        if i > 0:
            ha_open[i] = (ha_open[i - 1] + ha_close[i-1]) / 2
            ha_high[i] = max(ha_open[i], ha_close[i], high[i])
            ha_low[i] = min(ha_open[i], ha_close[i], low[i])
            
    return ha_open, ha_high, ha_low, ha_close

def transform_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None:
        df.open, df.high, df.low, df.close = build_ha(
            df.open.to_numpy(), df.high.to_numpy(), 
            df.low.to_numpy(), df.close.to_numpy()
        )
    return df

# -----------------------------------------------------------------------------
def get_data():
    h = Hermes(exchange='binance')
    
    symbol = 'BTCUSDT'
    interval = '1h'
    start = -1000
    end = 'one hour ago UTC'
    
    res = h.get_ohlcv(symbols=symbol, interval=interval, start=start, end=end)
    
    if res.get('success'):
        return res['message']
    
# @execution_time
def test_heikin_ashi(df, runs=1):
    for _ in range(runs):
        return transform_to_heikin_ashi(df)

# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    df = get_data()
    
    if df is not None:
        df = transform_to_heikin_ashi(df)
        c = BasicChart(df=df, color_scheme='day')
        c.draw()
        
        
    
    
    
    
    
    
    
    
