#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""

import time
import sys
import pandas as pd

from analysis.indicators import Indicators

# ==============================================================================
class Shakti:
    """Shakti is our Position Size Manager"""
    
    def __init__(self):
        
        pass
    
    # --------------------------------------------------------------------------
    def get_max_position_size(self, atr:float, portfolio_value:float, 
                              risk_amount:float) -> float:
        
        return risk_amount / atr
    
    def get_position_size_by_volatility(self, df:pd.DataFrame, 
                                        risk_amount:float) -> pd.Series:
        
        # at first we need the block value ... how much profit/loss 
        # would result from a 1% change in price
        block_value = df['close'] / 100
        
        # we also need the average change in price over the last
        # <lookback> period
        lookback = 21
        returns = abs(df['close'].pct_change() * 100)
        avg_returns = returns.ewm(span=lookback).mean()
        
        # now we can determine the risk of holding one unit of the
        # instrument ('instrument currency volatility') 
        instrument_currency_vol = block_value * avg_returns 

        pos_size = risk_amount / instrument_currency_vol
        # pos_size = pos_size / df['close']
        
        # # ----------------------------------------------------------------------
        # d = {'close' : df['close'], 'block value' : block_value, 'returns' : returns,
        #      'avg returns' : avg_returns, 'instr_cur_vol' : instrument_currency_vol,
        #      'p.size' : pos_size}
        # print_df = pd.DataFrame(data=d)
        # print(print_df.tail(40))
        # sys.exit()

        return pos_size
        
        
    def get_add_to_position(self, current_position_size:float, entry_price:float,
                            last_price:float, new_entry_price:float) -> float:
        """Calculates the base quantity to buy to reach the given average
        (new) entry price.

        :param current_position_size: size (base quantity) of current position 
        :type current_position_size: float
        :param entry_price: (average) entry price of current position 
        :type entry_price: float
        :param last_price: last/current price
        :type last_price: float
        :param new_entry_price: new average entry price after adding to position 
        :type new_entry_price: float
        :return: base qty to buy to reach new_entry_price 
        :rtype: float
        """
        
        if new_entry_price < entry_price:
            return 0 
        
        ps = current_position_size
        ep = entry_price
        lp = last_price
        nep = new_entry_price

        x = (ps * ep - nep * ps) / (nep - lp)         
        
        return x   
        
    def calculate_leverage(self, quote_asset_amount:float, 
                           position_value:float) -> float:
        
        if quote_asset_amount >= 0:
            leverage = 1
        else:
            debt = abs(quote_asset_amount)
            gross = position_value - debt
            leverage = round(debt / gross + 1, 2)
            
            # print(f'{gross} / {position_value} = {leverage=} ... ({debt=})')
            
        return leverage
        
    
    # --------------------------------------------------------------------------
    # helper methods
    def _add_atr_column(self, df:pd.DataFrame) -> pd.DataFrame:
        
        if 'atr' in df.columns:
            return df
        
        i = Indicators()
        return i.average_true_range(df=df)