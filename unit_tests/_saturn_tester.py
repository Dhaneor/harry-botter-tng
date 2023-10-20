#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work
import sys
import os
import numpy as np
from termcolor import cprint
from random import choice, random, randrange

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from staff.saturn import Saturn
from models.symbol import Symbol
from exchange.binance_ import Binance
from exchange.kucoin_ import Kucoin
from broker.models.orders import MarketOrder, LimitOrder, Order, StopMarketOrder, StopLimitOrder
from helpers.timeops import execution_time

from pprint import pprint

# =============================================================================
class OrderFactory:

    def __init__(self, symbol):

        self.symbol = symbol
        
        if '-' in symbol.symbol_name:
            with Kucoin() as conn:
                res = conn.get_ticker(symbol.symbol_name)
        else:        
            with Binance() as conn:
                res = conn.get_ticker(symbol.symbol_name)

        if res['success']:
            ticker = res.get('message', {})
            self.last_price = float(ticker.get('last', 0.00))
            print(self.last_price)
            print('-'*80)
        else:
            pprint(res)
            sys.exit()

    def create_market_order(self, side:str=None, base_qty:float=None, 
                            quote_qty:float=None) -> MarketOrder:

        if side is None:
            side = choice(['BUY', 'SELL'])
            
        auto_borrow = True if random() > 0.7 else False

        return MarketOrder(symbol=self.symbol,
                            market='SPOT',
                            side=side,
                            base_qty=base_qty,
                            quote_qty=quote_qty,
                            last_price=self.last_price,
                            auto_borrow=auto_borrow)

    def create_stop_limit_order(self, side:str='SELL', quantity:float=0.00
                                ) -> StopLimitOrder:

        if side == 'SELL':
            stop_price = self.last_price * 0.99
            limit_price = self.last_price * 0.98
        elif side == 'BUY':
            stop_price = self.last_price * 1.01
            limit_price = self.last_price * 1.02

        return StopLimitOrder(symbol=self.symbol,
                              market='SPOT',
                              side=side,
                              base_qty=quantity,
                              stop_price=stop_price,
                              limit_price=limit_price,
                              last_price=self.last_price)


    def create_multiple_market_orders(self, side:str, quantities:list) -> list:

        orders = []

        for q in quantities:
            self.create_market_order(side=side, quantity=q)

        return orders


# =============================================================================
class OrderChecker:

    separator = '*'*135

    def __init__(self, symbol : Symbol):

        self.symbol = symbol
        self.saturn = Saturn(symbol=self.symbol)


    # -------------------------------------------------------------------------
    def check_single(self, order : Order = None):

        return self.saturn.cm.check_order(order)

    # -------------------------------------------------------------------------
    def check_multiple(self, orders : list) -> list:

        for o in orders:
            print(self.separator)
            print(o)
            o = self.saturn.cm.check_order(o)
            print(0)

        return orders

# =============================================================================
class TestSaturn:

    separator = '*'*135

    def __init__(self, symbol_name : str):

        self.symbol = Symbol(symbol_name)
        self.factory = OrderFactory(symbol=self.symbol)
        self.checker = OrderChecker(symbol=self.symbol)

    # -------------------------------------------------------------------------
    @execution_time
    def test_market_orders(self, count:int=5):
        base_max = self.symbol.f_marketLotSize_maxQty
        base_min = self.symbol.f_marketLotSize_minQty
        base_qty_over_max = base_max * (1 + random())
        base_qty_below_min = base_min * random()
        valid_qtys = np.random.uniform(base_min, base_max, count-2)

        quantities = [base_qty_below_min, base_qty_over_max, *valid_qtys]

        for base_qty in quantities:
            
            quote_qty = base_qty * self.factory.last_price if random() > 0.5 else None
            
            # choose a randomized combination of base_qty and quote_qty to 
            # simulate cases were one (=allowed) or none/both (=not allowed)
            # are used
            _r = random()
            if _r < 0.025:
                base_qty = None
                quote_qty = None
            elif _r < 0.5:
                quote_qty = None
            elif _r < 0.9:
                base_qty = None 
            
            print(f'{base_qty=} :: {quote_qty=}')
            # create and check order
            order = self.factory.create_market_order(base_qty=base_qty, 
                                                     quote_qty=quote_qty)
            print(order)
            order = self.checker.check_single(order=order)

            # print color-coded result
            if order.status == 'APPROVED':
                cprint(order, 'green')
            else:
                cprint(order, 'red')
            print(TestSaturn.separator)

    @execution_time
    def test_stop_limit_orders(self, count:int=3):

        orders, quantities = [], ['0.0', 0.0, 0.3143098132143, 1, 10, 100]

        for q in quantities:
            order = self.factory.create_stop_limit_order(quantity=q)
            orders.append(order)
            print(order)
            order = self.checker.check_single(order=order)
            
            if order.status == 'APPROVED':
                cprint(order, 'green')
            else:
                cprint(order, 'red')




# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    
    symbol_name = 'XRP-BTC'

    ts = TestSaturn(symbol_name=symbol_name)

    ts.test_market_orders(8)
    # ts.test_stop_limit_orders()

