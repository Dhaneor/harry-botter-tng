#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 27 14:010:23 2022

@author dhaneor
"""
from random import random, choice

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------
from broker.util.order_factory import OrderFactory
from broker.models.symbol import Symbol
from util.timeops import execution_time
from broker.ganesh import Ganesh

BROKER = Ganesh(exchange='kucoin', credentials={}, market='CROSS MARGIN')
OF = OrderFactory()
# =============================================================================
def get_last_price(symbol):
    return BROKER.get_last_price(symbol)

def get_random_base_qty():
    factor = choice([0.01, 0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
    base_qty = random() * factor
    base_qty = round(base_qty, 8)
    return base_qty if random() < 0.85 else None

def get_random_quote_qty():
    quote_qty = 0.2 * random() if random() < 0.3 else 10_000 * random()
    quote_qty = round(quote_qty, 8)
    return quote_qty if random() < 0.15 else None

def get_random_limit_price(last_price):
    factor = random() - 0.3
    limit_price = round(last_price * factor, 8)
    return limit_price if random() < 0.5 else None

def get_random_stop_price(limit_price):
    factor = 0.95
    if limit_price:
        return choice([limit_price * factor, limit_price / factor])
    else:
        return None

# ..............................................................................
# @execution_time
def test_build_buy_order(symbol:Symbol, last_price:float):

    for _ in range(1):
        order = OF.build_buy_order(
            symbol=symbol,
            base_qty=get_random_base_qty(), #type:ignore
            last_price=last_price,
            type=choice(['limit', 'market']),
            limit_price=get_random_limit_price(last_price)) #type:ignore

        # print(order)


# @execution_time
def test_build_sell_order(symbol:Symbol, last_price:float):

    for _ in range(10):
        order = OF.build_sell_order(
            symbol=symbol,
            base_qty=get_random_base_qty(),
            last_price=last_price,
            type=choice(['limit', 'market']),
            limit_price=get_random_limit_price(last_price))

        # print(order)

@execution_time
def test_build_long_stop_order(symbol:Symbol, last_price:float):

    for _ in range(10):
        limit_price = get_random_limit_price(last_price)

        order = OF.build_long_stop_order(
            symbol=symbol,
            base_qty=get_random_base_qty(), #type:ignore
            stop_price=get_random_stop_price(limit_price), #type:ignore
            limit_price=limit_price,
            last_price=last_price)

        print(order)

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    symbol_name = 'BTC-USDT'
    symbol = BROKER.get_symbol(symbol_name)
    if symbol:
        test_build_buy_order(symbol, get_last_price(symbol.name))