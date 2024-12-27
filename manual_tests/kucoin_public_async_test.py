#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 05 20:02:20 2022

@author dhaneor
"""
import os, sys
import time
import asyncio
from pprint import pprint

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.exchange.kucoin_async import KucoinCrossMargin
from util.timeops import execution_time

conn = KucoinCrossMargin()

# ------------------------------------------------------------------------------

async def test_get_server_time():
    return await conn.get_server_time()

async def test_get_server_status():
    return await conn.get_server_status()

async def test_get_currencies():
    return await conn.get_currencies()

async def test_get_markets():
    return await conn.get_markets()

async def test_get_symbols():
    return await conn.get_symbols()

async def test_get_all_tickers():
    return await conn.get_all_tickers()

async def main(func):
    res = (
        await asyncio.gather(
            # test_get_server_time(),
            # test_get_server_status(),
            # test_get_currencies(),
            # test_get_markets(),
            # test_get_symbols(),
            test_get_all_tickers()
        )
    )

    for r in res:
            if isinstance(r, (str, int, float)):
                print(r)
            elif isinstance(r, (list, tuple)):
                pprint(r[0])
            elif isinstance(r, dict):
                print(r)





# ------------------------------------------------------------------------------
if __name__ == '__main__':
    st = time.time()
    func = test_get_server_status

    asyncio.run(main(func))

    et = round((time.time() - st) * 1000, 1)
    print(f'execution time: {et}ms')
