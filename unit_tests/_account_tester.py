#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import asyncio
import sys
import os
import time
import logging
from typing import Iterable
import pandas as pd

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

from src.broker.ganesh import Ganesh
from src.broker.models.async_account import Account
from broker.config import CREDENTIALS

broker = Ganesh(
    exchange='kucoin', market='cross margin', credentials=CREDENTIALS
    )

quote_asset='USDT'

# =============================================================================
async def start_account():
    acc = Account(broker=broker, quote_asset=quote_asset)
    await acc.run()
    return acc

async def run_account():
    acc = None
    try:
        acc = Account(broker=broker, quote_asset=quote_asset)
        await acc.run()
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        if acc:
            asyncio.gather(acc.stop())

async def test_get_account():
    acc = Account(broker=broker, quote_asset=quote_asset)
    await acc.initialize_from_exchange_api()
    [print(asset, pos) for asset, pos in acc.positions.items() if pos.value > 0]
    await asyncio.sleep(0.1)


    print('='*80)
    # print(acc.get_account_value(quote_asset=quote_asset))

def test_get_all_positions(include_dust:bool):
    [print(pos) for pos in acc.get_all_positions(include_dust)]

def test_reset_account(dont_touch: Iterable[str]):
    acc.reset_account(dont_touch=dont_touch)

def test_debt_ratio():
    print(acc.debt_ratio)

def test_account_leverage():
    print(acc.account_leverage)

def test_print_account():
    print(acc)

def test_find_ghost_positions():
    active_assets = ['XRP', 'ADA']
    quote_asset = 'USDT'
    current_positions = acc.get_all_positions(include_dust=False)
    current_positions = [
        pos for pos in current_positions if pos.asset != quote_asset
    ]

    [print(pos) for pos in current_positions]

    if current_positions:
        acc.reset_account(dont_touch=active_assets)

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    func = run_account
    # func = start_account
    # func = test_get_account

    # test_reset_account(())

    # test_get_all_positions(False)
    # test_reset_account(dont_touch=['ADA', 'ETH'])
    # test_debt_ratio()
    # test_account_leverage()
    # test_print_account()

    # test_find_ghost_positions()

    asyncio.run(func())
