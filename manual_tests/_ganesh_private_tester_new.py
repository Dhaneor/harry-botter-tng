#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import logging
import time
import random, string
from pprint import pprint
from typing import List

# ----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.broker.ganesh import Ganesh
from src.broker.util.order_factory import OrderFactory
from util.timeops import execution_time
from broker.config import CREDENTIALS

# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

GANESH = Ganesh(exchange='kucoin', market='CROSS MARGIN', credentials=CREDENTIALS)
ORDERFACTORY = OrderFactory()

# =============================================================================
# BUY/SELL
def _get_order(symbol_name:str, side:str, amount:float):
    if side == 'BUY':
        build_function = ORDERFACTORY.build_buy_order
        base_qty, quote_qty = 90, None
    elif side == 'SELL':
        build_function = ORDERFACTORY.build_sell_order
        base_qty = 10 / GANESH.get_last_price(symbol_name)
        quote_qty = None
    else:
        return

    symbol = GANESH.get_symbol(symbol_name=symbol_name)

    if symbol:
        return build_function(
            symbol=symbol,
            base_qty=base_qty,
            quote_qty=quote_qty,
            last_price=GANESH.get_last_price(symbol_name)
        )

# .............................................................................
def test_check_if_we_need_to_borrow(symbol_name:str, side:str, amount:float):

    symbol = GANESH.get_symbol(symbol_name)

    if not symbol:
        return

    order = _get_order(symbol_name=symbol_name, side=side, amount=amount)

    if not order:
        return

    need_to_borrow = GANESH._check_if_we_need_to_borrow(order)
    res = '' if need_to_borrow else 'DO NOT'

    if symbol is not None and order is not None:
        asset = symbol.quote_asset if order.side == 'BUY' else symbol.base_asset

        print(order)
        print(GANESH.get_balance(asset))
        print(F'We {res} need to borrow')

def test_market_buy(symbol_name:str, side:str, amount:float):
    symbol = GANESH.get_symbol(symbol_name)
    if not symbol:
        return

    print(GANESH.get_balance(symbol.base_asset))
    print(GANESH.get_balance(symbol.quote_asset))

    order = _get_order(symbol_name=symbol_name, side=side, amount=amount)

    if not order:
        return

    print(order)

    # .........................................................................
    order = GANESH.execute(order)

    print(order)
    print(GANESH.get_balance(symbol.base_asset))
    print(GANESH.get_balance(symbol.quote_asset))

def test_market_sell(symbol_name:str, side:str, amount:float):
    symbol = GANESH.get_symbol(symbol_name)
    if not symbol:
        return

    print(GANESH.get_balance(symbol.base_asset))
    print(GANESH.get_balance(symbol.quote_asset))

    order = _get_order(symbol_name=symbol_name, side=side, amount=amount)
    if not order:
        return

    print(order)

    # .........................................................................
    order = GANESH.execute(order)

    print(order)
    print(GANESH.get_balance(symbol.base_asset))
    print(GANESH.get_balance(symbol.quote_asset))


# -----------------------------------------------------------------------------
# everything with ORDERS
def test_get_active_stop_orders():
    orders = GANESH.get_active_stop_orders()
    if orders:
        [pprint(o.__dict__) for o in orders]
    else:
        print('no active orders found')

# -----------------------------------------------------------------------------
# BORROW/REPAY
def _show_overview(asset):
    info = GANESH.get_margin_loan_info(asset)
    ov = f'balance: {info.total_balance} / liabilty: {info.liability} / '
    ov += f'max borrow size: {info.max_borrow_size}'
    print(ov)

@execution_time
def test_get_margin_risk_limit():
    pprint(GANESH.margin_risk_limit)

    """
    {
        'borrowMaxAmount': 700.0,
        'buyMaxAmount': 750.0,
        'currency': 'MOVR',
        'holdMaxAmount': 6875.3,
        'precision': 8.0
    }
    """

def test_get_margin_configuration():
    pprint(GANESH.margin_configuration)

    """
    {
        'currencyList': [
            'XEM',
            'MATIC',
            'VRA',

            'XMR',
            'ZIL'],
        'liqDebtRatio': '0.97',
        'maxLeverage': 5,
        'warningDebtRatio': '0.95'
    }
    """

def test_get_borrow_details(asset:str=''):
    if asset:
        pprint(GANESH.get_borrow_details_for_asset(asset))
    else:
        pprint(GANESH.borrow_details)

    """
    {
        'availableBalance': 0.0,
        'currency': 'ZIL',
        'holdBalance': 0.0,
        'liability': 0.0,
        'maxBorrowSize': 1090.0,
        'totalBalance': 0.0
    }
    """

@execution_time
def test_get_margin_loan_info(assets: List[str]):
    for asset in assets:
        pprint(GANESH.get_margin_loan_info(asset).__dict__)
        print('-'*80)

@execution_time
def test_borrow():
    asset, amount = 'USDT', 10
    _show_overview(asset)

    res = GANESH.borrow(asset=asset, amount=amount)
    print('-'*80)
    pprint(res)
    print('-'*80)

    time.sleep(1)

    _show_overview(asset)

def test_repay():
    asset = 'USDT'
    mli = GANESH.get_margin_loan_info(asset)

    print(mli)

    amount = mli.liability
    available = mli.available_balance

    if amount > available:
        amount = available

    if amount > 0:
        res = GANESH.repay(asset=asset, amount=amount)
        pprint(res)
    else:
        print('nothing to repay')

    print('-'*80)
    _show_overview(asset)






# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    # test_check_if_we_need_to_borrow(
    #     symbol_name='XLM-USDT', side='SELL', amount=150
    # )

    # test_market_buy(symbol_name='XLM-USDT', side='BUY', amount=150)
    # time.sleep(1)
    # print('~_=•=_~'*25)
    # test_market_sell(symbol_name='XLM-USDT', side='SELL', amount=150)

    # -------------------------------------------------------------------------
    test_get_active_stop_orders()

    # -------------------------------------------------------------------------
    # test_get_margin_risk_limit()
    # test_get_margin_configuration()
    # test_get_borrow_details()
    # test_get_margin_loan_info(['USDT'])

    # test_borrow()
    # test_repay()