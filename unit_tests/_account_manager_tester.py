#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import time
import logging
import pandas as pd
from multiprocessing import Queue

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

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

from staff.account_manager import AccountManager
from models.users import Account
from staff.hermes import Hermes
from analysis.oracle import LiveOracle
from helpers.timeops import execution_time
from config import ACCOUNTS

hermes = Hermes(exchange='kucoin', mode='live')
oracle = LiveOracle()
q = Queue()
global am
am = None

# =============================================================================
class FakeOhlcvObserver:
    
    def register_subscriber(*args, **kwargs):
        id = kwargs.get('id')
        LOGGER.info(f'registered subscriber {id}')
        return

item = ACCOUNTS['one']    
account = Account(
    name='test', 
    exchange=item.get('exchange'),
    market='cross margin',
    quote_asset='USDT',
    is_active=item.get('is active'),
    api_key=item['credentials'].get('api_key'),
    api_secret=item['credentials'].get('api_secret'),
    api_passphrase=item['credentials'].get('api_key'),
    date_created=int(time.time()),
    risk_level=item.get('risk level'),
    max_leverage=item.get('max leverage'),
    user_id=item.get('user id', -1),
    # strategies=[item.get('strategy')]
)

def start_account_manager():
    oo = FakeOhlcvObserver()
    
    global am
    am = AccountManager(
        account=account, ohlcv_observer=oo, oracle=oracle, # type: ignore
        notify_queue=q
    )

@execution_time
def test_handle_ohlcv_update(data: list):
    
    msg = {
        'id': 'None(test)',
        'data' : data,
    }
    
    if am:
        am.handle_ohlcv_update(msg)
        
@execution_time
def test_get_max_leverage_allowed_by_exchange():
    if am is not None:
        ml = am._get_max_leverage_allowed_by_exchange()
        print(ml, type(ml))
    
# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    symbols = ['XRP-USDT'] #, 'ADA-USDT']
    interval = '1m'
    start = -1000
    end = 'now UTC'
    data = []
    
    for symbol in symbols:
        res = hermes.get_ohlcv(
            symbols=symbol, interval=interval, start=start, end=end
        )
        df = res.get('message')
        
        if isinstance(df, pd.DataFrame):
            df['s.all'] = -1
            
            data.append( 
                {
                    'symbol': symbol,
                    'interval': interval,
                    'data': df
                }
            )
    
    
    start_account_manager()
    test_handle_ohlcv_update(data=data)
    # test_get_max_leverage_allowed_by_exchange()
    
    