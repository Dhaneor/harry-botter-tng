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
from random import randint

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from config import CREDENTIALS
from exchange.exchange import ExchangePublic as Broker
from helpers.timeops import (seconds_to, unix_to_utc, execution_time,
                             get_start_and_end_timestamp,
                             interval_to_milliseconds)

BROKER = Broker(exchange='kucoin')
print('Successfully started the broker')

# ==============================================================================
@execution_time
def test_initialize_broker(market:str, mode:str):

    broker = None
    
    try:
        broker = Broker('kucoin')
    except (ValueError, TypeError) as e:
        print(e) 
    except Exception as e:
        print(e)
        
    if broker is not None:
        print(f'{broker.name} initialized: True')
        client = True if broker.broker.client else False
        print(f'client initialized: {client}' )
    else:
        print('Could not initialize broker ...')
        
@execution_time
def test_get_timestamp_and_status():
    
    print('getting server time and status:', end='')

    res_ts = BROKER.get_server_time()
    res_status = BROKER.get_server_status()
    
    print('OK')
        
    if res_ts.get('success'):
        server_time = res_ts['message']  
    else: 
        server_time = 0
        pprint(res_ts)
    
    if res_status.get('success'):
        # pprint(res_status['message'])
        status = res_status['message']['status']  
    else: 
        status = 'no connection'
        
    print(f'{BROKER.name} STATUS at {unix_to_utc(server_time)}:\t{status}')
            
@execution_time
def test_get_currencies():
    with Broker('kucoin', 'CROSS MARGIN') as conn:
        res = conn.get_currencies()
      
    if res['success']: 
        currencies = res['message']
        pprint(currencies)
        print('-'*80)
        print(f'there are {len(currencies)} currencies/assets on the exchange')
        sys.exit()
        margin_currencies = [cur['fullName'] for cur in currencies \
            if cur['isMarginEnabled']]
        
        print(f'Found {len(margin_currencies)} Kucoin Margin currencies')
        
    else:
        pprint(res)

@execution_time
def test_get_ohlcv(symbol:str, interval:str, start:int=None, end:int=None):
    _st = time()
    with Broker('kucoin', 'CROSS MARGIN') as conn:
        et = round((time() - _st)*1000)
        print(f'establishing connection to {conn.name} took {et} ms')
        res = conn.get_ohlcv(symbol=symbol, interval=interval, 
                             start=start, end=end, as_dataframe=True)
        
    if res['success']: 

        if isinstance(res['message'], pd.DataFrame):
            df = res['message']
            print(df)
            # print(df.info())  
        else:
            _t, _l = type(res['message']), len(res['message'])
            print(f'result: {_t} with {_l} elements')
            # pprint(res['message'])
        
        print('-'*160)
        print(f"execution time for api call: {seconds_to(res['execution time']/1000)} \
                ({res['execution time']})")
    else:  
        pprint(res)

@execution_time
def test_get_earliest_valid_timestamp(symbol:str, interval:str='1d'):
    with Broker('kucoin', 'CROSS MARGIN') as conn:
        res = conn._get_earliest_valid_timestamp(symbol, interval)
    
    if res['success']:
        ts = res['message']  
        print(f'earliest timestamp for {symbol} is {ts} ({unix_to_utc(ts)})')
    else:
        pprint(res)

@execution_time                
def test_get_markets():
    res = BROKER.get_markets()
        
    if res['success']:
        pprint(res['message'])
    else:
        pprint(res)

@execution_time            
def test_get_symbols(quote_asset=None, runs=1):
    
    for _ in range(runs): 
        message = BROKER.get_symbols(quote_asset=quote_asset)       
        test = [item['symbol'] for item in message]
        print(test)
        print(len(test))
        
    return message 
        
@execution_time  
def test_get_symbol(symbol:str=None):
    pprint(BROKER.get_symbol(symbol))

@execution_time
def test_get_ticker(symbol:str=None):
    try: 
        pprint(BROKER.get_ticker(symbol))
    except ValueError as e:
        print(e)
        

@execution_time
def test_get_all_tickers(market:str='SPOT'):
    tickers = BROKER.get_all_tickers()
    print(f'found {len(tickers)} tickers ({type(tickers)})')
    return tickers

@execution_time
def test_get_risk_limits(runs=1):
    for _ in range(runs):
        risk_limits = BROKER.get_margin_risk_limit()
        elem = randint(0, len(risk_limits))
        pprint(risk_limits[elem])

# .............................................................................
@execution_time
def get_chunks(start:str, end:str, interval:str) -> None:
    start, end = get_start_and_end_timestamp(start=start, end=end,
                                             unit='milliseconds', 
                                             interval=interval, verbose=True)
    
    with Kucoin() as conn:
        chunks = conn._get_chunk_periods(start=start, end=end, interval=interval)
        latency = conn._get_latency()
        
    pprint(chunks)
    print(f'number of chunks: {len(chunks)}')
    print(f'latency: {latency}')
    
def test_check_too_many_requests():
    counter, _st = 0, time()
    with Broker('kucoin', 'CROSS MARGIN') as conn:
        while True:
            counter += 1
            res = conn.get_server_status()
            code = str(res.get('status code'))
            print(counter, code)
            if code == '429':
                et = round(time() - _st)
                break
        
    print(f'\nlimit on KUCOIN exceeded after {counter} requests in {et} seconds')

def test_binance_wrapper(runs=10):
    with Binance() as conn:
        for _ in range(runs):
            r = conn._test_wrapper()
            if r is not None:
                if r.get('success'):
                    print('r is: ', r['message'])
                else:
                    print('r is: ', r)
            else:
                print('r is: None')

@execution_time
def test_get_latency(runs=3):
    with Broker('kucoin', 'CROSS MARGIN') as conn:
        print(f'average latency: {conn.get_latency(runs=runs)}ms')
                
# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    symbol = 'QNTA-USDT'
    interval = '12h'
    start = '2018-06-26 00:00:00'
    end = '2022-06-27 00:00:01'

    # .........................................................................
    # test_initialize_broker(market='CROSS MARGIN', mode='live')
    # test_get_timestamp_and_status()
    # test_get_latency(runs=5)  
    # test_get_markets()

    # test_get_currencies()
    test_get_symbols(quote_asset='BTC', runs=1)   
    # test_get_symbol(symbol=symbol)
    
    # test_get_ticker(symbol)
    # test_get_all_tickers()
    
    # test_get_risk_limits(runs=5)
    
    # .........................................................................    
    # test_get_ohlcv(symbol=symbol, interval=interval, start=start, end=end)
    # test_get_earliest_valid_timestamp(symbol)
    
    # .........................................................................
    # get_chunks(start, end, interval)
    # test_check_too_many_requests()
    # test_binance_wrapper(runs=1200)



