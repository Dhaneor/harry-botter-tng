#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import logging
import random, string
from pprint import pprint

# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from broker.jupiter import Jupiter
from broker.models.requests import RequestFactory
from models.users import Users, Accounts
from util.timeops import execution_time
from broker.config import CREDENTIALS

# ------------------------------------------------------------------------------
LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

JUPITER = None

# ------------------------------------------------------------------------------
@execution_time
def init_jupiter():
    global JUPITER

    JUPITER = Jupiter(
        exchange='Kucoin', market='CROSS MARGIN', user_account=CREDENTIALS
        )

def get_target_account():
    return [
        {'asset' : 'XLM',
         'target' : 0.00,
         'stop_loss' : [[0.10, 0.4],
                        [0.09, 0.6]
                        ],
         'take_profit' : [[0.15, 1]],
         'quote_asset' : 'USDT'
         },
        {'asset' : 'XRP',
         'target' : 13.00,
         'stop_loss' : [[0.42, 0.5],
                        [0.40, 0.5]
                        ],
         'take_profit' : None,
         'quote_asset' : 'USDT'
         },
        {'asset' : 'ADA',
         'target' : 0.00,
         'stop_loss' : [[0.253, 0.5],
                        [0.2, 0.5]
                        ],
         'take_profit' : None,
         'quote_asset' : 'USDT'
         },
        {'asset' : 'GRT',
         'target' : 0.00,
         'stop_loss' : [[0.08, 0.5],
                        [0.06, 0.5]
                        ],
         'take_profit' : None,
         'quote_asset' : 'USDT'
         },
        ]

def get_assets(real:bool=True):

    def _get_fake_assets(number_of_assets:int):
        length = random.choice([3, 4])
        letters = string.ascii_uppercase
        return [''.join(random.choice(letters) for i in range(length))
                for _ in range(number_of_assets)
                ]

    real_assets = ['BTC', 'ETH', 'LTC', 'XMR', 'UNI', 'XLM', 'XRP', 'ADA']

    if real:
        return ['BTC', 'ETH', 'LTC', 'XMR', 'UNI', 'XLM', 'XRP', 'ADA']
    else:
        return real_assets + _get_fake_assets(1000)


# ..............................................................................
@execution_time
def test_request_factory():
    t = get_target_account()

    assets = get_assets()

    r = RequestFactory()
    requests = r.convert_target_account_to_requests(t, all_assets=assets)

    pprint(requests)

@execution_time
def test_build_requests():

    @execution_time
    def _build_them():
        return JUPITER._build_requests(t)

    init_jupiter()
    t = get_target_account()

    # _build_them()

    pprint(_build_them())


@execution_time
def test_update_account():

    target_account = get_target_account()

    pprint(target_account)
    print('-~*~-'*30)
    print('\n')


    JUPITER.update_account(target_account)


# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    init_jupiter()
    test_update_account()