#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:05:23 2021

@author_ dhaneor
"""

import symbol
import sys
import os
import time

from pprint import pprint
from typing import Union
from random import choice

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from broker.ganesh import Ganesh
from broker.models.exchange_order import build_exchange_order

from staff.saturn import Saturn
from models.symbol import Symbol
from broker.models.orders import MarketOrder, StopLimitOrder, StopMarketOrder
from util.timeops import execution_time

from broker.config import CREDENTIALS

# =============================================================================
class GaneshTester:

    def __init__(self, symbol:str, side:str='LONG'):

        st = time.time()
        self.name = 'GaneshTester'

        self.exchange = 'kucoin' if '-' in symbol else 'binance'
        self.market = 'SPOT' if self.exchange == 'binance' else 'CROSS MARGIN'
        self.symbol = Symbol(symbol)
        self.side = side

        self._last_price = 0.00

        self.ganesh = Ganesh(
            exchange='kucoin', market=self.market, credentials=CREDENTIALS
            )
        self.saturn = Saturn(symbol=self.symbol, verbose=True)

        self.hline = '\n' + '-=*=-' * 30 + '\n'

        et = round((time.time() - st)*1000)
        print(f'\n[{self.name}] startup time: {et}ms\n')
        return

    # -------------------------------------------------------------------------
    @property
    def last_price(self):
        if self._last_price == 0:
            self._last_price = self.ganesh._get_last_price()
        return self._last_price

    @property
    def min_position_size(self):
        if self.exchange == 'binance':
            return self.symbol.f_minNotional_minNotional / self.last_price * 1.05
        else:
            return 5 / self.last_price

    def balance_free(self, asset:str):
        res = self.ganesh.get_balance(asset)

        if res['success']:
            return res['message']['free']
        else:
            pprint(res)
            return 0.00

    # -------------------------------------------------------------------------
    def test_check_latency(self):
        [print(f'latency {self.ganesh.check_latency()} ms') for _ in range(10)]

    @execution_time
    def test_system_status(self):
        print(self.ganesh.system_status)

    @execution_time
    def test_get_account(self, non_zero:bool=True):
        try:
            pprint(self.ganesh.account)
        except Exception as e:
            print(e)

    @execution_time
    def test_get_margin_loan_info(self):
        data = self.ganesh.margin_loan_info
        for _ in range(5):
            [print(item) for item in data]
            print(f'\nfound {len(data)} borrowable assets')

    @execution_time
    def test_get_margin_configuration(self):
        pprint(self.ganesh.margin_configuration)

    @execution_time
    def test_get_fees(self, symbols:Union[list, str, None]=None):

        if symbols is None:
            symbols = [choice(self.ganesh.valid_symbols) for _ in range(10)]

        res = self.ganesh.get_fees(symbols)

        if res.get('success'):
            message = res.get('message')
            [print(item) for item in message]
        else:
            pprint(res)

    @execution_time
    def test_get_balance(self):

        for _ in range(50):
            pprint(
                self.ganesh.get_balance(
                    choice(self.ganesh.valid_assets)
                    )
                )
            print(' ')


    @execution_time
    def get_order_status(self, order_id:str):
        res = self.ganesh.get_order_status(order_id)
        pprint(res)


    @execution_time
    def test_get_orders(self):
        orders = self.ganesh.orders

        if not orders:
            print('found no orders at all! :(')
            return

        if self.ganesh._return_orders_as_exchange_order_objects:
            [print(item) for item in orders]
        else:
            [print(build_exchange_order(item)) for item in orders]

    # -------------------------------------------------------------------------
    @execution_time
    def test_market_buy(self):
        print('-'*160)

        order = MarketOrder(symbol=self.symbol,
                            market=self.market,
                            side='BUY',
                            base_qty=self.min_position_size,
                            last_price=self.last_price,
                            auto_borrow=True)

        print(order)
        order = self.saturn.cm.check_order(order)
        print(order)
        if order.status == 'APPROVED':
            self.ganesh.execute(order)
        print(order)

        pprint(self.ganesh.get_balance(asset=self.symbol.baseAsset))
        print('-~*~-'*30)

    @execution_time
    def test_market_sell(self):
        print('-'*160)

        order = MarketOrder(symbol=self.symbol,
                            market=self.market,
                            side='SELL',
                            base_qty=self.balance_free(self.symbol.baseAsset),
                            last_price=self.last_price)

        print(order)
        order = self.saturn.cm.check_order(order)
        print(order)

        if not order.status == 'REJECTED':
            order = self.ganesh.execute(order)

        print(order)

        pprint(self.ganesh.get_balance(asset=self.symbol.baseAsset))
        print('-~*~-'*30)

    # .........................................................................
    @execution_time
    def test_panic_sell(self):
        res = self.ganesh.panic_sell()

        print('-'*80)
        if res['success']:
            pprint(res['message'])
            base_asset = self.symbol.baseAsset

            asset_left_in_account = self.balance_free(base_asset)
            print('-'*80)
            print(f'we now have {asset_left_in_account} {base_asset} left')
        else:
            pprint(res)

    # .........................................................................
    @execution_time
    def test_stop_order(self, side:str='LONG', type:str='market',
                        execute:bool=True) -> Union[str, None]:

        if side == 'LONG':
            stop_price = self.last_price * 0.9
            limit_price = self.last_price * 0.85
            side = 'SELL'

        print('-'*160)

        if type == 'market':
            order = StopMarketOrder(symbol=self.symbol,
                                    market=self.market,
                                    side=side,
                                    base_qty=self.balance_free(self.symbol.baseAsset),
                                    stop_price=stop_price,
                                    last_price=self.last_price)

        else:
            order = StopLimitOrder(symbol=self.symbol,
                                market=self.market,
                                side=side,
                                base_qty=self.balance_free(self.symbol.baseAsset),
                                stop_price=self.last_price * 0.9,
                                limit_price=limit_price,
                                last_price=self.last_price)

        print(order)
        self.saturn.cm.check_order(order)
        print(order)

        if not order.status == 'REJECTED':
            order = self.ganesh.execute(order)
            print(order)

            if not order.status == 'FAILED':
                return order.result.get('orderId')

        return None

    # .........................................................................
    @execution_time
    def test_cancel(self, order_id:str):

        res = self.ganesh.cancel(order_id)
        pprint(res)

    @execution_time
    def test_cancel_all(self):
        res = self.ganesh.cancel_all()
        pprint(res)

    @execution_time
    def test_cancel_stop_and_get_order_result(self, order_id:str=None):
        res = self.ganesh.cancel_stop_loss_and_get_order_result(order_id=order_id)
        pprint(res)


    # .........................................................................
    # combined tests
    def test_create_and_cancel_stop_order(self, type='market'):
        order_id = self.test_stop_order(type=type)
        time.sleep(3)
        self.test_cancel_stop_and_get_order_result(order_id)



# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    symbol = 'ALGO-USDT'

    gt = GaneshTester(symbol=symbol)
    # -------------------------------------------------------------------------
    # gt.test_check_latency()
    # gt.test_system_status()
    # gt.test_get_balance()
    # gt.test_get_account()
    # gt.test_get_margin_loan_info()
    # gt.test_get_margin_configuration()

    # gt.test_get_fees(symbols=None)


    # gt.get_order_status('2971814603')
    # gt.test_get_orders()

    sys.exit()

    gt.test_market_buy()
    gt.test_market_sell()

    gt.test_stop_order()
    # gt.test_stop_order()
    # gt.test_stop_order()

    time.sleep(7)

    # gt.test_cancel_stop_and_get_order_result()
    # gt.test_cancel('2971814603')
    gt.test_cancel_all()

    gt.test_market_sell()

    # gt.test_panic_sell()

    # .........................................................................
    # gt.test_create_and_cancel_stop_order()
    # time.sleep(7)
    # gt.test_market_sell()



