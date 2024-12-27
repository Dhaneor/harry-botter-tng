#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 05 20:02:20 2022

@author dhaneor
"""
import os, sys
import time
import asyncio
import random
from pprint import pprint

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.exchange.kucoin_async import KucoinCrossMargin
from util.timeops import execution_time
from broker.config import CREDENTIALS

conn = KucoinCrossMargin(credentials=CREDENTIALS)

# =============================================================================
async def test_get_margin_config():
    res = await conn.get_margin_config()
    pprint(res)

async def test_get_account():
    res = await conn.get_account()
    pprint(res)

async def test_get_orders():
    res = await conn.get_orders()
    print(f'got {len(res)} orders') if res else print('nothing')

async def test_get_active_orders():
    res = await conn.get_active_orders()

async def test_get_active_stop_orders():
    res = await conn.get_active_stop_orders()

async def test_get_multiple_orders():
    try:
        res = await conn._get_multiple_orders(
            symbol='XRP-USDT',
            side=None,
            start=None, # 'November 01, 00:00:00 2022',
            end=None, # 'December 05, 00:00:00 2022'
        )
        print(f'got {len(res)} orders')
    except Exception as e:
        print(e)


async def test_buy_market():
    try:
        res = await conn.buy_market(
            symbol='XRP-USDT',
            base_qty=10
        )
        pprint(res)
    except Exception as e:
        print(e)

async def test_sell_market():
    try:
        res = await conn.sell_market(
            symbol='XRP-USDT',
            base_qty=10
        )
        pprint(res)
    except Exception as e:
        print(e)

async def test_buy_limit():
    try:
        res = await conn.buy_limit(
            symbol='XRP-USDT',
            base_qty='10',
            price='0.35'
        )
        pprint(res)
    except Exception as e:
        print(e)

async def test_sell_limit():
    try:
        res = await conn.sell_limit(
            symbol='XRP-USDT',
            base_qty='9',
            price='0.8'
        )
        pprint(res)
    except Exception as e:
        print(e)

async def test_stop_limit():
    try:
        res = await conn.stop_limit(
            symbol='XRP-USDT',
            side='SELL',
            base_qty='9',
            stop_price='0.33',
            limit_price='0.32'
        )
        pprint(res)

        orders = await conn.get_active_stop_orders()
        pprint(orders)
    except Exception as e:
        print(e)

async def test_stop_market():
    try:
        tasks = []
        for _ in range(10):
            kwargs = dict(
                symbol='XRP-USDT',
                side='SELL',
                base_qty='1.5',
                stop_price=str(round(0.34 + (random.random() - 0.5) / 20, 4)),
            )

            tasks.append(asyncio.create_task(conn.stop_market(**kwargs)))

        res = await asyncio.gather(*tasks)
        pprint(res)
    except Exception as e:
        print(e)

async def test_cancel_order():
    try:
        orders = await conn.get_active_orders()
        pprint(orders)
        if orders:
            for order in orders:
                res = await conn.cancel_order(order['orderId'])
                print(res)
    except Exception as e:
        print(e)

async def test_cancel_all_orders():
    try:
        orders = await conn.get_active_stop_orders()
        pprint(orders)
        if orders:
            res = await conn.cancel_all_orders(symbol='XRP-USDT')
            print(res)
    except Exception as e:
        print(e)


async def test_get_margin_risk_limit():
    try:
        res = await conn.get_margin_risk_limit()
        pprint(res)
    except Exception as e:
        print(e)

async def test_get_borrow_details():
    try:
        res = await conn.get_borrow_details(asset='XLM')
        pprint(res)
    except Exception as e:
        print(e)

async def test_get_liability():
    try:
        res = await conn.get_liability()
        pprint(res)
    except Exception as e:
        print(e)


# =============================================================================
if __name__ == '__main__':

    # func = test_get_margin_config
    # func = test_get_account

    # func = test_get_orders
    # func = test_get_active_orders
    # func = test_get_active_stop_orders
    # func = test_get_multiple_orders

    # func = test_buy_market
    # func = test_sell_market
    # func = test_buy_limit
    # func = test_sell_limit

    # func = test_stop_limit
    # func = test_stop_market

    # func = test_cancel_order
    # func = test_cancel_all_orders

    # func = test_get_margin_risk_limit
    # func = test_get_borrow_details
    func = test_get_liability

    asyncio.run(func())