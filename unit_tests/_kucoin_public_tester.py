#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import sys
import os
from time import time
import pandas as pd
from pprint import pprint
from typing import Optional, Tuple

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from src.exchange.kucoin_ import KucoinCrossMargin, is_exchange_closed
from src.helpers.timeops import (unix_to_utc, execution_time,
                             get_start_and_end_timestamp,
                             interval_to_milliseconds)
from config import CREDENTIALS

conn = KucoinCrossMargin(credentials=CREDENTIALS)

# ==============================================================================
@execution_time
def _test_kucoin_emergency_server_status():
    print(f'Exchange is closed/in maintanence: {is_exchange_closed()}')
      
@execution_time
def test_get_timestamp_and_status():

    try:
        timestamp = conn.get_server_time()
        status = conn.get_server_status()
    except:
        timestamp = 0
        status = 'no connection'         
    finally:
        print(timestamp, status)
            
@execution_time
def test_get_currencies():
    currencies = conn.get_currencies()
    
    if not currencies:
        return

    print('-'*80)
    print(f'there are {len(currencies)} currencies/assets on the exchange')
    margin_currencies = [cur['fullName'] for cur in currencies \
        if cur['isMarginEnabled']]
    
    print(f'Found {len(margin_currencies)} Kucoin Margin currencies')

@execution_time
def test_get_ohlcv(symbol:str, interval:str, start: Optional[int]=None, 
                   end: Optional[int]=None):
    res = conn.get_ohlcv(symbol=symbol, interval=interval, 
                         start=start, end=end, as_dataframe=True)
    
    if not res['success']:
        pprint(res)
        return
        
    if isinstance(res['message'], pd.DataFrame):
        df = res['message']
        print(df)
    else:
        _t, _l = type(res['message']), len(res['message'])
        print(res['message'])
        print(f'result: {_t} with {_l} elements')


@execution_time
def test_get_earliest_valid_timestamp(symbol:str, interval:str='1d'):
    ts = conn._get_earliest_valid_timestamp(symbol, interval)    
    print(f'earliest timestamp for {symbol} is {ts} ({unix_to_utc(ts)})')


@execution_time                
def test_get_markets():
    pprint(conn.get_markets())

@execution_time            
def test_get_symbols(quote_asset=None):
    symbols = conn.get_symbols(quote_asset=quote_asset)  
    pprint(symbols[0])        
    print(f'got {len(symbols)} symbols')

        
@execution_time  
def test_get_symbol(symbol: str):
    pprint(conn.get_symbol(symbol))

@execution_time
def test_get_ticker(symbol:str):
    pprint(conn.get_ticker(symbol=symbol))

@execution_time
def test_get_all_tickers(quote_asset: str=''):
    tickers = conn.get_all_tickers()
    
    if tickers and quote_asset:
        all_symbols = conn.get_symbols(quote_asset=quote_asset)
        symbol_names = [item['symbol'] for item in all_symbols]
        tickers = [t for t in tickers if t.get('symbol') in symbol_names]
 
    if tickers:
        print(f'found {len(tickers)} tickers')
        return tickers
              
# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    symbol = 'ETH-USDT'
    interval = '1d'
    start = '2019-07-01 00:00:00'
    end = '2019-12-31 00:00:00'
    
    # .........................................................................
    # _test_kucoin_emergency_server_status()
    
    # test_get_timestamp_and_status() 
    # test_get_markets()

    for _ in range(1):
        # test_get_currencies()
        test_get_symbols(quote_asset='BTC')    
        # test_get_symbol(symbol=symbol)
    
    # test_get_ticker(symbol=symbol)
    # test_get_all_tickers(quote_asset='')
    
    # .........................................................................    
    # test_get_ohlcv(symbol=symbol, interval=interval, start=start, end=end)
    # test_get_earliest_valid_timestamp(symbol)
    
    # .........................................................................
    # get_chunks(start, end, interval)
    # test_check_too_many_requests()
    # test_binance_wrapper(runs=1200)


