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
from collections import namedtuple

# ----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from broker.ganesh import Ganesh
from broker.position_handlers.workers import (PositionUpdateRequest,
                                              PositionWorker, LoanRequest,
                                              LoanWorker)
from util.timeops import execution_time
from broker.config import CREDENTIALS

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

GANESH = Ganesh(exchange='kucoin', market='CROSS MARGIN', credentials=CREDENTIALS)
SYMBOL = GANESH.get_symbol('XRP-USDT')
LOANWORKER = LoanWorker(broker=GANESH, symbol=SYMBOL)
POSITIONWORKER = PositionWorker(broker=GANESH, symbol=SYMBOL)

# =============================================================================
@execution_time
def _execute(request):
    if isinstance(request, PositionUpdateRequest):
        POSITIONWORKER.execute_request(request)
    elif isinstance(request, LoanRequest):
        LOANWORKER.execute_request(request)

def _get_amount_to_open_position(target_size:float):
    balance = GANESH.get_balance(SYMBOL.base_asset)
    free = balance.get('net')

    return (target_size - free)

def _get_amount_to_close_position():
    balance = GANESH.get_balance(SYMBOL.base_asset)
    return balance.get('free')


# =============================================================================
def test_loan_worker():

    asset = SYMBOL.base_asset
    amount = 5.00

    LOGGER.info(GANESH.get_balance(asset))

    request = LoanRequest(action='borrow', asset=asset, amount=amount)
    _execute(request)

    balance = GANESH.get_balance(asset)
    LOGGER.info(balance)

    time.sleep(3)

    # .........................................................................
    borrowed = balance.get('borrowed')
    request = LoanRequest(action='repay', asset=asset, amount=borrowed)
    _execute(request)

    LOGGER.info(GANESH.get_balance(asset))

def test_position_worker(target_size:float):
    asset = SYMBOL.base_asset
    amount = _get_amount_to_open_position(target_size)

    if amount > 0:
        side = 'BUY'
    elif amount <  0:
        side = 'SELL'
    else:
        LOGGER.info('no position change required')
        return

    LOGGER.info(GANESH.get_balance(asset))
    LOGGER.info(f'going to {side} {amount} {asset}')

    request = PositionUpdateRequest(side=side, asset=asset, base_qty=amount)

    _execute(request)
    LOGGER.info(GANESH.get_balance(asset))

# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    test_position_worker(target_size=0)

    # test_loan_worker()
