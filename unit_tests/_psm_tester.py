#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 11  12:20:23 2021

@author: dhaneor
"""
import sys
import os
import time
import pandas as pd
from pprint import pprint
from random import random
# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------

from staff.shakti import Shakti
from staff.hermes import Hermes

from helpers.timeops import execution_time


# ==============================================================================
class ShaktiTester:
    
    def __init__(self):
        
        self.shakti = Shakti()
        self.df : pd.DataFrame = None # ohlcv dataframe
        self.capital = 1000 # how much capital do we have
        self.risk_amount = 100 # how much are we willing to risk for the trade

    # --------------------------------------------------------------------------        
    def get_ohlcv(self, symbol:str, interval:str, start:str, end:str):
        
        hermes = Hermes(exchange='binance')
        
        res = hermes.get_ohlcv(
            symbols=symbol, interval=interval, start=start, end=end
            )
        
        if res.get('success'):
            self.df = res.get('message')
        
        assert self.df is not None
                   
    def add_atr(self):
        
        if self.df is not None:
            self.df = self.shakti._add_atr_column(self.df)
            
    def get_max_position_size(self):
        
        if self.df is not None:
            self.df['p.size.max'] = self.df.apply(
                lambda x: self.shakti.get_max_position_size(
                    x['atr'], self.capital, self.risk_amount
                    ),
                axis=1
                )
            
            self.df['quote_qty_max'] = self.df['p.size.max'] * self.df['close']
            self.df['leverage_max'] = self.df['quote_qty_max'] / self.capital
             
    def get_add_to_position(self):
        
        current_position_size = 10
        entry_price = 100
        last_price = 150
        new_entry_price = last_price * 0.8
        precision = 8
        
        buy_qty = self.shakti.get_add_to_position(current_position_size,
                                                  entry_price,
                                                  last_price,
                                                  new_entry_price)
        buy_qty = round(buy_qty, precision)
        
        new_position_size = current_position_size + buy_qty
        new_position_size = round(new_position_size, precision)
        
        print(f'we buy {buy_qty} and then we have {new_position_size}')
        
        check = current_position_size * entry_price + buy_qty * last_price 
        check /= new_position_size
            
        print(check)
            
        
        
        
# ==============================================================================
@execution_time
def test_get_ohlcv(symbol, interval, start, end):
    
    st = ShaktiTester()
    st.get_ohlcv(symbol, interval, start, end)
    
@execution_time
def test_add_atr(symbol, interval, start, end):
    
    st = ShaktiTester()
    st.get_ohlcv(symbol, interval, start, end)
    st.add_atr()
    print(st.df)

@execution_time
def test_get_max_position_size(symbol, interval, start, end):
    
    st = ShaktiTester()
    st.get_ohlcv(symbol, interval, start, end)
    st.add_atr()
    st.get_max_position_size()
    print(st.df)
    
def test_get_add_to_position(symbol, interval, start, end):
    
    st = ShaktiTester()
    # st.get_ohlcv(symbol, interval, start, end)
    st.get_add_to_position()
    
    
# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    symbol = 'XRPUSDT'
    interval = '4h'
    start, end = 'January 01, 2022 00:00:00', 'March 15, 2022 00:00:00'
    
    # test_get_add_to_position(symbol, interval, start, end)

    test_get_max_position_size(symbol, interval, start, end)

        
