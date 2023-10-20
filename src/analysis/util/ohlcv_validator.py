#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23 11:07:23 2022

@author dhaneor
"""
import sys
import pandas as pd
import numpy as np

from helpers.timeops import (interval_to_milliseconds, unix_to_utc,
                             get_start_and_end_timestamp)


# =============================================================================
class OhlcvValidator:
    
    def __init__(self):
        pass
    
    # -------------------------------------------------------------------------
    def find_missing_rows_in_df(self, df:pd.DataFrame, interval:str, start, end) -> list:

        start_ts, end_ts = get_start_and_end_timestamp(
            start=start, end=end, interval=interval
        )

        i = interval_to_milliseconds(interval)
        res = []
        
        first_open_time = df.iloc[0].loc['open time']
        if first_open_time > (start_ts + i - 1):
            res.append((unix_to_utc(start_ts), unix_to_utc(first_open_time)))    

        # .....................................................................
        df['_diff'] = df['open time'].diff() # - df['open time'].shift()
        df['res_s'] = np.nan
        df['res_e'] = np.nan
        
        df.loc[(df['_diff'] != i), 'res_s'] = df['open time'].shift()
        df.loc[(df['_diff'] != i), 'res_e'] = df['open time']
        df.at[0, 'res_e'] = np.nan
        
        _tmp = list(zip(
            df['res_s'].dropna().tolist(), df['res_e'].dropna().tolist() 
        ))
        
        _tmp = [(unix_to_utc(item[0]), unix_to_utc(item[1])) for item in _tmp]
                
        res += _tmp
        df.drop(['_diff', 'res_s', 'res_e'], axis=1, inplace=True)
        
        # .....................................................................
        last_open_time = df.iloc[-1].loc['open time']
        if (last_open_time + i) < end_ts:
            res.append((unix_to_utc(last_open_time), unix_to_utc(end_ts))) 
        
        return res