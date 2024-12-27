#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
from time import time
from pprint import pprint

# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from broker.config import API_KEY, API_SECRET, API_PASSPHRASE

from broker.models.account import Account
from broker.models.position import (Position, LongPosition, ShortPosition,
                                    PositionDetective, LongPositionDetective,
                                    ShortPositionDetective)
from broker.models.balance import balance_factory
from broker.position_handlers.position_handler import *
from broker.ganesh import Ganesh
from util.timeops import execution_time


BROKER = None
ACCOUNT = None
CREDENTIALS = {'api_key' : API_KEY,
               'api_secret' : API_SECRET,
               'api_passphrase' : API_PASSPHRASE
               }


# -----------------------------------------------------------------------------
@execution_time
def initialize_globals():
    global BROKER, ACCOUNT
    BROKER = Ganesh(
        exchange='kucoin', market='CROSS MARGIN', credentials=CREDENTIALS
        )
    ACCOUNT = Account(BROKER)

# -----------------------------------------------------------------------------
def test_get_account():
    initialize_globals()

    for pos in ACCOUNT.positions:
        print(pos)
        if not pos.asset in BROKER.valid_assets:
            print(f'{pos.asset} is not a valid asset!')


@execution_time
def test_balance_creation():
    initialize_globals()

    non_zero = [bal for bal in BROKER.account \
        if any(arg for arg in (bal['total'] != 0,
                               bal['borrowed'] != 0)
               )
        ]

    [print(balance_factory(bal)) for bal in non_zero]

@execution_time
def test_initialize_from_exchange_api():
    initialize_globals()

    for _ in range(1):

        ACCOUNT.initialize_from_exchange_api()
        [print(pos) for pos in ACCOUNT.positions] # if pos.type is not None]
        print('='*120)
        print('\n')
        pprint(ACCOUNT.get_position('ALGO').position_detective.active_stop_orders)
        pprint(ACCOUNT.get_position('ALGO').position_detective.active_limit_orders)

def test_position_detective(asset:str):
    initialize_globals()
    ACCOUNT.initialize_from_exchange_api()

    p = ACCOUNT.get_position(asset)
    p.position_detective.do_your_thing()
    print(p)



# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    pass
    # test_get_account()
    # test_balance_creation()

    # --------------------------------------------------------------------------
    # test_initialize_from_exchange_api()
    # test_position_detective('XRP')