import sys
import os
import time

from pprint import pprint
from typing import Union
from random import random, randint, choice
import logging

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
from broker.position_handlers.position_handler import *
from broker.models.balance import Balance, balance_factory
from broker.models.requests import PositionChangeRequest, RequestFactory
from models.symbol import Symbol
from broker.util.order_factory import OrderFactory
from broker.models.exchange_order import build_exchange_order
from util.timeops import execution_time

from broker.config import CREDENTIALS

BROKER: Ganesh

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to LOGGER
LOGGER.addHandler(ch)


# =============================================================================
def test_logger():
    LOGGER.debug('debug message')
    LOGGER.info('info message')

def initialize_broker():
    global BROKER
    BROKER = Ganesh(
        exchange='kucoin', market='CROSS MARGIN', credentials=CREDENTIALS
    )
    print(BROKER)

def _get_target_account(id_=1):

    ta = [
        {'asset' : 'XRP',
         'target' : 13.0928726,
         'stop_loss' : [(0.7, 0.5), (0.8, 0.5)],
         'take_profit' : None,
         'quote_asset' : 'USDT'
         },
        {'asset' : 'XMR',
         'target' : 0.00,
         'stop_loss' : None,
         'take_profit' : None,
         'quote_asset' : 'USDT'
         }
    ]

    return ta[id_ - 1]

def _get_position_change_request():
    target_account = _get_target_account(id_=1)
    asset = target_account['asset']
    quote_asset = target_account['quote_asset']

    rf = RequestFactory(quote_asset=quote_asset)
    return rf._build_request_object(asset, target_account)

def create_tp_orders(symbol: Symbol, no_of_orders: int=3):

    def _get_order_amounts(asset_balance):
        amounts = []
        for _ in range(no_of_orders - 1):
            amounts.append(max(
               asset_balance * random() / (no_of_orders - 1),
               symbol.f_lotSize_minQty * 5
            ))

        amounts.append(asset_balance - sum(amounts))

        return amounts

    # .........................................................................
    balance = BROKER.get_balance(symbol.baseAsset) # type:ignore
    if balance:
        amount = balance.get('free')

    if amount < symbol.f_minNotional_minNotional:
        raise Exception('we don´t have the money, boss!')

    order_amounts = _get_order_amounts(amount)
    print(f'{sum(order_amounts)} -> {order_amounts}')

    last_price = BROKER.get_last_price(symbol.name)
    order_factory = OrderFactory()

    for amount in order_amounts:

        price = last_price + random() * last_price

        order = order_factory.build_sell_order(
            symbol=symbol, type='LIMIT', base_qty=amount, limit_price=price,
            last_price = last_price
            )

        print(order)

        if order.status == 'APPROVED':
            BROKER.execute(order)

# .............................................................................
@execution_time
def test_update_stop_loss_handler():

    @execution_time
    def _execute():
        h = UpdateStopLossHandler(broker=BROKER, symbol=symbol)
        h.execute()

    # ..........................................................................
    initialize_broker()
    chosen_one = choice(BROKER.get_active_stop_orders())
    pprint(build_exchange_order(chosen_one))

    symbol = BROKER.get_symbol(chosen_one.get('symbol')) # type:ignore

    print(symbol)

    try:
        create_tp_orders(symbol,no_of_orders=3)
    except Exception  as e:
        print(f'unable to create test tp orders, because: {e}')

    _execute()

@execution_time
def test_action_factory():

    @execution_time
    def _get_action(runs):
        for _ in range(runs):
            action = af.get_action(symbol=symbol, balance=balance, request=request)
        return action

    initialize_broker()


    request = _get_position_change_request()
    balance = balance_factory(BROKER.get_balance(request.asset)) # type:ignore

    symbol_name = f'{request.asset}-{request.quote_asset}'
    symbol = BROKER.get_symbol(symbol_name)

    LOGGER.info(balance)
    LOGGER.info(request)
    LOGGER.info(symbol)

    af = ActionFactory(broker=BROKER)
    _ = af.get_action(symbol=symbol, balance=balance, request=request)
    action = _get_action(runs=100)

    if action:
        pprint(action.__dict__)
    else:
        print(action)


# ------------------------------------------------------------------------------
#                                   MAIN                                       #
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # test_update_stop_loss_handler()

    test_action_factory()