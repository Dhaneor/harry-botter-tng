#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 15:05:23 2021

@author_ dhaneor
"""
import sys
import os
import time
import logging

from pprint import pprint
from typing import Union
from random import random, randint, choice

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

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
import exchange.util.repositories as r
from exchange.util.repositories import (AccountRepository, OrderRepository,
                                      TickerRepository, SymbolRepository,
                                      RiskLimitRepository, 
                                      BorrowDetailsRepository,
                                      MarginConfigurationRepository,
                                      PublicRepository, PrivateRepository,
                                      event_bus
                                      )
from exchange.util.event_bus import (LoanRepaidEvent, OrderCreatedEvent, 
                                     OrderFilledEvent, OrderCancelledEvent)
from helpers.timeops import execution_time
from config import CREDENTIALS

EXCHANGE = Kucoin(market='CROSS MARGIN', credentials=CREDENTIALS)

EVENTS = [
    OrderCreatedEvent, OrderFilledEvent, OrderCancelledEvent, LoanRepaidEvent
    ]

# ==============================================================================
def test_order_repository():
    
    orep = OrderRepository(client=EXCHANGE)
    
    while True:
        try:
            # orders = orep.orders
            # pprint(build_exchange_order(orders[randint(0, len(orders)-1)]))

            time.sleep(2)
            
            r.event_bus.publish_event(choice(EVENTS)())
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    
def test_account_repository():
    
    arep = AccountRepository(client=EXCHANGE)
    
    while True:
        try:
            account = arep.account
            pprint(account[randint(0, len(account)-1)])
            time.sleep(5)
            
            if random() > 0.3:
                EVENT_BUS.publish_event(choice(EVENTS))
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)

def test_ticker_repository():
    
    trep = TickerRepository(client=EXCHANGE)
    
    while True:
        try:
            pprint(trep.tickers[-1])
            time.sleep(15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)
            
def test_symbol_repository():
    
    srep = SymbolRepository(client=EXCHANGE)
    
    while True:
        try:
            pprint(srep.symbols[-1])
            time.sleep(15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)
            
def test_risk_limit_repository():
    
    rlrep = RiskLimitRepository(client=EXCHANGE)
    
    while True:
        try:
            pprint(rlrep.risk_limits[randint(0, 150)])
            time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)
            
def test_borrow_details_repository():
    
    bdrep = BorrowDetailsRepository(client=EXCHANGE)
    
    while True:
        try:
            pprint(bdrep.borrow_details[randint(0, 150)])
            print(f'DEBT RATIO: {bdrep.debt_ratio}')
            time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)

def test_margin_configuration_repository():
    mcrep = MarginConfigurationRepository(client=EXCHANGE)
    
    while True:
        try:
            pprint(mcrep.margin_configuration)
            time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    

@execution_time
def test_public_repository():
    r = PublicRepository(client=EXCHANGE)
    
    print([s['symbol'] for s in r.symbols][:10])
    print('*'*150)
    pprint(r.__dict__)
    
@execution_time
def test_private_repository():
    prep = PrivateRepository(client=EXCHANGE)
    
    while True:
        print(prep.orders[-1])
        time.sleep(1)
        
        prep.notify(OrderCancelledEvent('abc'))
        
        time.sleep(3)

    
# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    test_private_repository()