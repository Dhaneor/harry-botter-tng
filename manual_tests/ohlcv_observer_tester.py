#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 01 14:03:23 2021

@author dhaneor
"""

from codecs import charmap_encode
import sys
import os
import time
import logging

from pprint import pprint
from typing import Iterable

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

from analysis.util.ohlcv_observer import (Chronos, TimeDifferenceWatcher,
                                          OhlcvObserver)
from util.timeops import execution_time

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# =============================================================================
def test_update_timer():
    def _callback(expired_intervals):
        str_ = ', '.join(expired_intervals)
        print(f'expired intervals: {str_}')
        print('~–=*=–~' * 10)

    ut = Chronos(_callback)
    ut.start()
    print('test')

def test_on_full_minute(minute:int):
    def _callback(value):
        print('>>>>', value)

    c = Chronos(callback=_callback)
    c.start()
    c._on_full_minute(0)

# =============================================================================
def test_time_difference_to_server():
    def on_update(time_difference):
        print(f'[on_update] time difference is {time_difference} ms')

    tdts = TimeDifferenceWatcher(on_update)
    tdts.run()

@execution_time
def test_ohlcv_observer():

    def callback(data:dict):
        for item in data['data']:
            symbol, interval = item['symbol'], item['interval']
            print('-'*80, f'\n{symbol}, {interval}:\n', item['data'].tail(10))
            LOGGER.info(f'processing {symbol}...')
            time.sleep(0.05)


    o = OhlcvObserver()

    o.register_subscriber(id='subby', symbols=['BTC-USDT', 'ETH-USDT'],
                          interval='1m', callback=callback)
    o.register_subscriber(id='another', symbols=['XRP-USDT', 'ETH-USDT'],
                          interval='1m', callback=callback)
    o.register_subscriber(id='third', symbols=['ADA-USDT', 'ETH-USDT'],
                          interval='5m', callback=callback)
    o.register_subscriber(id='fourth', symbols=['XMR-USDT', 'ETH-USDT'],
                          interval='15m', callback=callback)
    o.register_subscriber(id='fifth', symbols=['XLM-USDT', 'ETH-USDT'],
                          interval='1h', callback=callback)
    o.register_subscriber(id='sixth', symbols=['XLM-USDT', 'ETH-USDT'],
                          interval='3m', callback=callback)

    # o.intervals_have_passed(['1h', '4h', '1d'])

@execution_time
def test_download_one_symbol():
    o = OhlcvObserver(exchange='kucoin')
    print(
        o._download_ohlcv_for_one_symbol('BTC-USDT', '1m')['data']
    )

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    # test_update_timer()
    # test_on_full_minute(0)

    # test_time_difference_to_server()
    test_ohlcv_observer()


    # disable the timers before running this one!
    # test_download_one_symbol()