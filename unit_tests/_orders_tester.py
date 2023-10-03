#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 19 22:35:27 2021

@author: dhaneor
"""

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

from broker.models.orders import *
from models.symbolos import Symbol
from staff.saturn import Saturn
from exchange.binance_ import Binance
from mock_responses.order_results import OrderResultConstructor

from pprint import pprint
from random import random
from termcolor import cprint

EXCHANGE = Binance

# =============================================================================
# helper functions
def get_random_quantity(symbol, quantity='valid'):

    factor_ = random()
    min_q = symbol.f_lotSize_minQty
    max_q = symbol.f_lotSize_maxQty
    
    if quantity == 'valid':
        return max(factor_ * max_q, min_q*1.01)    
    
    elif quantity == 'too big':
        return max_q / factor_

    elif quantity == 'too small':
        return min_q * factor_
        

def get_last_price(symbol):

    with EXCHANGE() as conn: 
        res = conn.get_ticker(symbol.symbol_name)
        if res['success']:
            price = float(res['message']['last'])
            return price

# =============================================================================
def test_create_order(symbol:Symbol, type:str=None, quantities:list=['valid']):

    last_price = get_last_price(symbol)
    line = '=-~•~-=' * 30

    for qty in quantities:
        
        line_color = 'red' if qty != 'valid' else 'green'
        cprint(line, line_color)
        
        if type == 'market':
        
            order = MarketOrder(symbol=symbol,
                                market='SPOT',
                                side='SELL' if random() < 0.5 else 'BUY',
                                base_qty=get_random_quantity(symbol, qty),
                                auto_borrow=True,
                                last_price=last_price
                                )
            
        if type == 'limit':
            
            order = LimitOrder(symbol=symbol,
                               market='SPOT',
                               side='SELL' if random() < 0.5 else 'BUY',
                               base_qty=get_random_quantity(symbol, qty),
                               limit_price=min((last_price * random() * 2), last_price),
                               last_price=last_price
                               )
            
            
        print('')
        print(order)

        saturn = Saturn(symbol=symbol)
        saturn.cm.check_order(order)

        print(order)

        if order.status != 'REJECTED':
            orc = OrderResultConstructor(symbol=symbol)
            order.execution_report = orc.execute(order=order)
            cprint(order, 'green')
        
        else:
            cprint(order, 'red')
        
        if order.status == 'FILLED':
            # pprint(order.last_execution_report.original_message['message'])
            print(f'real price: {order.real_price}\n\n')
            




# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    symbol = Symbol('XRPBTC')
    quantities = ['valid', 'too big', 'too small', 'valid', 'valid']
    # quantities = ['valid']

    print('\n\n\n\n')
    print('=-~•~-='*30)

    test_create_order(symbol=symbol, type='market', quantities=quantities)
