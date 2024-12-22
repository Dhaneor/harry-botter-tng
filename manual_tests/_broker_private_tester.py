#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import sys
import os
from time import sleep
from pprint import pprint
from typing import Union
from uuid import uuid1
from random import choice, random
import logging

LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

from exchange.exchange import Exchange as Broker
from models.symbol import Symbol
from broker.models.exchange_order import build_exchange_order
from util.timeops import seconds_to, unix_to_utc, execution_time
from broker.config import CREDENTIALS

EXCHANGE = 'kucoin'
MARKET = 'CROSS MARGIN'
BROKER = Broker(exchange=EXCHANGE, market=MARKET, credentials=CREDENTIALS)


# ------------------------------------------------------------------------------
def print_order(order):
    type = order['type']
    while len(type) < 18:
        type += ' '
    side = 'BUY ' if order['side'] == 'BUY' else 'SELL'
    symbol = order['symbol']
    amount = float(order['origQty'])
    if amount == 0:
        amount = float(order['executedQty'])

    if order['price'] is not None:
        price = float(order['price'])
    else:
        price = float(order['stopPrice'])

    if price == 0:
        price = float(order['cummulativeQuoteQty']) / float(order['executedQty'])

    funds = float(order['origQty']) * price
    limit_price = price
    stop = 'STOP' if float(order['stopPrice']) != 0 else '    '
    status = 'active' if order['isWorking'] else 'done'
    status = 'cancelled' if order['status'] == 'CANCELED' else status
    date_time = unix_to_utc(order['time'])

    print(f'[{date_time}] {symbol}  \t{amount:.8f} @ {price:.8f} \
        {stop} {type} {side} order [{status}]')

# -----------------------------------------------------------------------------
# ACCOUNT related
@execution_time
def test_get_account(interesting:list=None, non_zero:bool=False):
    pprint(BROKER.account)

@execution_time
def test_get_accounts(currency:str=None):
    BROKER.private.get_accounts(currency=currency, account_type='margin')

    if res['success']:
        accounts = res['message']
        acc_non_zero = [cur for cur in accounts if not cur['balance']=='0']
        pprint(acc_non_zero)
        print('-'*80)
    else:
        pprint(res)

@execution_time
def test_get_account_by_id(id_:str):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_account_by_id(id_)

    if res['success']:
        pprint(res['message'])
        print('-'*80)
    else:
        pprint(res)

@execution_time
def test_get_withdrawal_quota(currency:str='BTC'):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_withdrawal_quotas(currency=currency)

    if res['success']: pprint(res['message'])
    else: pprint(res)

@execution_time
def test_get_balance(asset:str):
    try:
        pprint(BROKER.private.get_balance(asset=asset))
    except ValueError as e:
        print(e)

@execution_time
def test_get_balance_by_currency(asset:str=None):
    res = BROKER.get_balance_by_currency(asset=asset)

    if res.get('success'):
        balance = res.get('message')
        pprint(balance)
    else:
        pprint(res)


@execution_time
def test_get_fees(symbol:str=None):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_fees(symbol=symbol)

        if res['success']:
            fees = res['message']
            pprint(fees)
        else:
            pprint(res)


# .............................................................................
# get ORDERs
@execution_time
def test_get_orders(symbol=None, side:str=None, order_type:str=None,
                    status:str=None, start=None, end=None):

    orders = BROKER.private.get_orders(symbol=symbol, side=side, order_type=order_type,
                               status=status, start=start, end=end)

    [print(build_exchange_order(o)) for o in orders]

@execution_time
def test_get_active_orders(symbol:str=None, side:str=None, order_type:str=None):
    orders = BROKER.get_active_orders(
        symbol=symbol, side=side, order_type=order_type
        )
    [print(build_exchange_order(o)) for o in orders]

@execution_time
def test_get_active_stop_orders(symbol:str=None, side=None):
    orders = BROKER.private.get_active_stop_orders(symbol=symbol, side=side)
    orders = [build_exchange_order(o) for o in orders]
    [print(o) for o in orders]
    return orders


@execution_time
def test_get_order(order_id:str=None, client_order_id:str=None):
    o = BROKER.private.get_order(order_id=order_id, client_order_id=client_order_id)

    pprint(o)

@execution_time
def test_get_fills(order_id:str=None, start:str=None):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_fills(order_id=order_id, start=start)

    if res['success']:
        pprint(res['message'])

# .............................................................................
# create ORDERS
@execution_time
def test_buy_market(symbol=None, base_qty=None, quote_qty=None,
                    base_or_quote='quote', auto_borrow=False) -> dict:

    client_order_id = str(uuid1())

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        if base_or_quote == 'base':
            res = conn.buy_market(symbol=symbol, client_order_id=client_order_id,
                                  base_qty=base_qty,auto_borrow=auto_borrow)
        else:
            res = conn.buy_market(symbol=symbol, client_order_id=client_order_id,
                                  quote_qty=quote_qty, auto_borrow=auto_borrow)

        pprint(res['message'])
        return res

@execution_time
def test_sell_market(symbol=None, base_qty=None, quote_qty=None,
                     base_or_quote='base', auto_borrow=False):

    client_order_id = str(uuid1())

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        if base_or_quote == 'base':
            res = conn.sell_market(symbol=symbol, client_order_id=client_order_id,
                                   base_qty=base_qty, auto_borrow=auto_borrow)
        else:
            res = conn.sell_market(symbol=symbol, client_order_id=client_order_id,
                                  quote_qty=quote_qty, auto_borrow=auto_borrow)

    pprint(res)

@execution_time
def test_buy_limit(symbol=None, base_qty=None, price=None, auto_borrow=False):

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:

        if price is None:
            res = conn.get_ticker(symbol=symbol)

        if res['success']:
            # levels = [0.9, 0.88, 0.87, 0.85, 0.83, 0.81]
            levels = [0.75]
            last_price = res['message']['last']
            prices = [round(float(last_price) * l, 3) for l in levels]
            print(prices)
        else:
            pprint(res)
            print('-'*80)
            print('unable to get price from exchange ... exiting')

        results = [conn.buy_limit(symbol=symbol,
                                  base_qty=base_qty,
                                  price=price,
                                  client_oid=uuid1()) for price in prices]

    for res in results:
        if res['success']:
            pprint(res['message'])
        else:
            pprint(res)


@execution_time
def test_sell_all(asset:str):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.sell_all(asset)
        pprint(res)
        print('~'*80)

        res = conn.get_account()
        if res['success']:
            accounts = res['message']
            balance = [acc for acc in accounts if acc['asset'] in symbol]
            print('-'*80)
            pprint(balance)

@execution_time
def test_create_stop_order(symbol:str=None, side:str='SELL', base_qty=None,
                           type:str='market', cancel:bool=True):

    s = Symbol(symbol)
    price_step = s.f_priceFilter_tickSize
    price_precision = s.f_tickPrecision
    base_step = s.f_lotSize_stepSize
    base_precision = s.f_stepPrecision
    last_price, balance_free, quote_precision = None, None, None

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        # get the latest price
        res = conn.get_ticker(symbol)
        if res['success']:
            last_price = float(res['message']['last'])
            pprint(res.get('message'))
            print('~'*80)
        else:
            pprint(res)
            return

        # get the current balance for the base asset
        balance = test_get_balance(asset=s.baseAsset)
        pprint(balance)
        balance_free = float(balance.get('free'))
        balance_borrowed = float(balance.get('borrowed'))

        # ......................................................................
        # determine the position size and the stop price
        if side == 'SELL':
            base_qty = round(balance_free, base_precision)
            base_qty = (base_qty - base_step) if base_qty > balance_free else base_qty
            stop_price = round(last_price * 0.95, price_precision)
        else:
            base_qty = round(
                balance_borrowed - balance_free + base_step, base_precision)
            stop_price = round(last_price / 0.95, price_precision)

        limit_price = None

        # ......................................................................
        # now create the order
        print(f'{base_qty=} :: {last_price} :: {stop_price=} :: {limit_price=}')

        if type == 'limit':
            limit_price = round(stop_price * 0.975, price_precision)
            res = conn.stop_limit(symbol=symbol, side=side, base_qty=base_qty,
                                stop_price=stop_price, limit_price=limit_price)
        elif type == 'market':
            res = conn.stop_market(symbol=symbol, side=side, base_qty=base_qty,
                                   stop_price=stop_price)

        sys.exit()
        print('-'*80)
        if res['success'] and cancel:
            order = res['message']
            pprint(order)

            sleep(1)
            res = conn.cancel_order(symbol=symbol,
                                    order_id=str(order['orderId']))

            print('-'*80)
            if res['success']:
                pprint(res['message'])
            else:
                pprint(res)
            print('-'*80)
        else:
            pprint(res)

    # test_get_active_stop_orders(symbol=symbol)
    test_get_balance(asset=base_asset)

# .............................................................................
# cancel ORDERS
@execution_time
def test_cancel_order(order_id:str):
    res = BROKER.private.cancel_order(order_id)

    pprint(res)

@execution_time
def test_cancel_all_orders(symbol=None):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.cancel_all_orders(symbol=symbol)

    if res['success']:
        number_deleted = len(res['message'])
        [pprint(item) for item in res['message']]
        print('-'*80)
        print(f'{number_deleted} orders cancelled ...')
    else:
        pprint(res)


# .............................................................................
# LOAN management
@execution_time
def test_get_margin_config():
    pprint(BROKER.private.get_margin_configuration())

@execution_time
def test_get_margin_risk_limit():
    pprint(BROKER.private.get_margin_risk_limit())

@execution_time
def test_get_borrow_details_for_all():
    pprint(BROKER.get_borrow_details_for_all())

@execution_time
def test_get_borrow_details(asset:str=None, runs=1):

    def get_it(asset):
        try:
            pprint(BROKER.get_borrow_details(asset))
        except ValueError as e:
            print(e)

    if not asset:
        res = BROKER.get_margin_configuration()

        if res.get('success'):
            assets = res['message']['currencyList']
        else:
            print(res.get('error'))
            sys.exit()

        for _ in range(runs):
            asset = choice(assets) if random() < 0.9 else 'ABCD'
            get_it(asset)
    else:
        get_it(asset)

@execution_time
def test_get_liability(asset:str=None):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_liability(asset=asset)

    if res['success']:
        pprint(res['message'])
    else:
        pprint(res)

@execution_time
def test_repay(asset:str=None):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_liability(asset=asset)

        if res['success']:
            orders = res['message']['items']

            if orders:

                liability = sum([float(o['liability']) for o in orders])

                res = conn.get_balance(asset=asset)

                if res['success']:
                    available = float(res['message']['availableBalance'])
                else: available = 0

                size = min(available, liability)
                print(f'going to repay {size} {asset} ...')

                if size > 0:
                    res = conn.repay(asset=asset,
                                        size=size)
                    pprint(res)

            else:
                print('nothing to repay. we are good!')
                print('-'*80)

        res = conn.get_balance(asset=asset)
        if res['success']:
            pprint(res['message'])


# .............................................................................
# order/fill response STANDARDIZATIOM
@execution_time
def test_fill_transformation():

    fill =  {'counterOrderId': '620848e0fec9a60001ec81f0',
             'createdAt': 1644710113000,
             'fee': '0.000499999995',
             'feeCurrency': 'USDT',
             'feeRate': '0.001',
             'forceTaker': True,
             'funds': '0.499999995',
             'liquidity': 'taker',
             'orderId': '620848e0fd31e50001a59db2',
             'price': '0.822',
             'side': 'buy',
             'size': '0.6082725',
             'stop': '',
             'symbol': 'XRP-USDT',
             'tradeId': '620848e02e113d325d83e24b',
             'tradeType': 'MARGIN_TRADE',
             'type': 'market'
            }

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn._standardize_fill_response([fill, fill, fill])

        pprint(res)

@execution_time
def test_standardize_order_response(order_id:str):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_order(order_id=order_id)

        if res['success']:
            response = res['message']
        else:
            pprint(res)
            return

        res = conn._standardize_order_response(response)
        pprint(res)

# -----------------------------------------------------------------------------
# COMBINED TESTS that use multiple calls
@execution_time
def test_open_long_position(symbol:str, quote_qty:Union[str, float]):
    res = test_buy_market(symbol=symbol, quote_qty=str(quote_qty),
                          auto_borrow=False)

    if res['success']:
        message = res['message']
        amount = float(message['dealSize'])
        spent = float(message['dealFunds']) # + float(message['fee'])
        price = spent / amount
        base_asset, quote_asset = symbol.split('-')[0],symbol.split('-')[1]
        print(f"Successfully bought {amount} {base_asset} \
            [price: {price} {quote_asset}]")

        test_create_stop_order(symbol)

        print('-'*80)
        # test_sell_all(asset=base_asset)

@execution_time
def test_open_short_position(symbol:str):
    """Opens a short position with 2x leverage.

    :param symbol: name of the symbol
    :type symbol: str
    """
    # find out, how much base currency we have/can borrow
    base_asset = symbol.split('-')[0]

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        base_details = conn.get_borrow_details(base_asset)
        ticker = conn.get_ticker(symbol)

        if base_details['success']:
            base_details = base_details['message']
        else:
            raise Exception(f'Unable to get borrow details for {base_asset}')

        if ticker['success']:
            ticker = ticker['message']
        else:
            raise Exception(f'Unable to get ticker for {symbol}')

    pprint(base_details)
    pprint(ticker)

    pos_size = base_details.get('availableBalance') \
                + base_details.get('maxBorrowSize') * 0.2

    s = Symbol(symbol_name=symbol)
    pos_size = round(pos_size, s.baseAssetPrecision)

    # ..........................................................................
    client_order_id = str(uuid1())

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.sell_market(symbol=symbol, client_order_id=client_order_id,
                                base_qty=pos_size, auto_borrow=True)
    if res.get('success'):
        message = res.get('message')
        pprint(message)
    else:
        pprint(res)
        sys.exit()

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_balance(asset=base_asset)

    if res.get('success'):
        message = res.get('message')
        free = float(message.get('free'))
        borrowed = float(message.get('borrowed'))

    sl_pos_size = borrowed - free
    print(f'{free=} : {borrowed=} -> {sl_pos_size=}')

@execution_time
def test_get_all_positions():

    def _is_empty(item):
        res = any(
            [float(item.get('total')) != 0, float(item.get('borrowed')) != 0]
            )

        print(item, float(item.get('total')), float(item.get('borrowed')), res)
        return res

    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_symbols(quote_asset='USDT')
        message = res.get('message') if res.get('success') else []
        symbols = [item['symbol'] for item in message]
        tokens = [item.split('-')[0] for item in symbols]

        res = conn.get_account()
        account = res.get('message') if res.get('success') else []
        account = [item for item in account if item.get('asset') in tokens\
            and any([float(item.get('total')) != 0,
                     float(item.get('borrowed')) != 0])]

        pprint(account)
        print(len(account))
        tokens = [item.get('asset') for item in account]

        tickers= conn.tickers
        tickers = [item for item in tickers \
            if item.get('symbol') in symbols \
            and item.get('symbol').split('-')[0] in tokens]

        print(f'found {len(tickers)} tickers')
        # pprint(tickers)

@execution_time
def test_get_filled_orders(symbol:str):
    with Broker(EXCHANGE, 'CROSS MARGIN') as conn:
        res = conn.get_orders(symbol=symbol, status='FILLED')

    if res['success']:
        orders = res['message']
        filled = [o for o in orders if o['status'] == 'FILLED']
        for o in filled:
            print('-'*80)
            print(unix_to_utc(o['updateTime']))
            pprint(o)

def test_order_repository_response():

    # def stop_order():
    #     BROKER.private.stop_market(symbol='XRP-USDT', )

    counter = 0

    while True:
        if counter == 0:
            try:
                o = test_get_active_stop_orders()[0]
            except:
                o = None

            if o:
                order_id = o.order_id

                print(f'will cancel order: {order_id}')

                test_cancel_order(order_id)

            counter = 1

        # .....................................
        sleep(10)

        try:
            o =test_get_active_stop_orders()[0]
        except:
            o = None

        # if o:
        #     order_id = o.order_id

        # print('='*120)
        # pprint(BROKER.private.exchange.get_orders()['message'][-1])
        # print('='*120)

def test_account_repository_response():

    order_id = ''

    def create_limit_order():
        res = BROKER.private.buy_limit(
            symbol='XRP-USDT', base_qty=10, price=0.35
        )
        if res['success']:
            message = res['message']
            global order_id
            order_id = message['orderId']
            print(f'created order with order_id: {order_id}')
        else:
            pprint(res)

    def cancel_order():
        global order_id
        print(f'cancelling order id {order_id}', end=' ')
        res = BROKER.private.cancel_order(order_id=order_id)
        if res['success']:
            print('OK')
            order_id = ''
        else:
            error = res['error']
            print(f'FAIL ({error})')

    def show_balance():
        account = BROKER.private.repository.account
        [print(item) for item in account if item['asset'] == 'USDT']


    # .........................................................................
    show_balance()

    create_limit_order()

    show_balance()

    sleep(5)
    cancel_order()

    show_balance()



@execution_time
def test_repay_for_single_order(asset:str=None):
    res = BROKER.get_liability(asset=asset)

    if res['success']:
        orders = res['message']['items']

        if orders:
            for o in orders:

                res = conn.get_balance(asset=asset)

                if res['success']:
                    available = float(res['message']['availableBalance'])
                else: available = 0

                size = min(available, float(o['liability']))
                print(f'going to repay {size} {asset} ...')

                if size > 0:
                    res = conn.repay(asset=asset,
                                    size=size,
                                    trade_id=o['tradeId'])
                    pprint(res)

        else:
            print('nothing to repay. we are good!')

    res = BROKER.get_balance(asset=asset)
    if res['success']:
        pprint(res['message'])

# -----------------------------------------------------------------------------
def test_public_is_singleton():
    another = Broker(exchange=EXCHANGE, market=MARKET, credentials=CREDENTIALS)

    print(BROKER.public, another.public)

    del another

# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    kucoin_symbol = 'XRP-USDT'

    if kucoin_symbol is not None:
        base_asset = kucoin_symbol.split('-')[0]
        quote_asset = kucoin_symbol.split('-')[1]
        binance_symbol = base_asset + quote_asset
        asset = base_asset
    else:
        binance_symbol, quote_asset, base_asset, asset = None, None, None, None

    if EXCHANGE == 'kucoin':
        symbol = kucoin_symbol
    else:
        symbol = binance_symbol

    interval = '12h'
    start = None #'February 02, 2022 00:00:00'
    end = None # 'now UTC'

    base_qty = 60.00
    quote_qty = 10.00

    # .........................................................................
    # ACCOUNT related methods
    #
    # test_get_accounts()
    # test_get_withdrawal_quota()
    # test_get_account()
    # test_get_account(non_zero=True)
    # test_get_balance(asset=base_asset)
    # test_get_balance(asset=base_asset)
    # test_get_balance_by_currency(asset=quote_asset)
    # test_get_fees(symbol=symbol)


    # .........................................................................
    # get ORDERS
    #
    # test_get_orders()
    # test_get_orders(symbol=symbol, order_type=None, side=None,
    #                 status=None, start=start, end=end)
    # print('-=*=-'*15)
    # test_get_active_orders(symbol=None, side=None, order_type=None)
    # test_get_active_stop_orders(symbol=None, side=None)

    # BINANCE orders
    # test_get_order(symbol=symbol, order_id='4037761293')

    # KUCOIN orders
    # test_get_order(symbol=symbol, order_id='620848e0fd31e50001a59db2') # MARKET BUY FILLED
    # test_get_order(symbol=symbol, order_id='620782247ee3b40001c73613') # LIMIT BUY FILLED
    # test_get_order('6206e1f4f235a400017ef74b') # LIMIT BUY CANCELLED
    # test_get_fills(order_id='620782247ee3b40001c73613')
    # test_get_fills(start='1644489741717')

    # test_get_filled_orders(symbol=symbol)
    # test_get_fills(order_id='620782247ee3b40001c73613')

    # test_fill_transformation()
    # test_standardize_order_response('6207822147ee3b40001c73613')
    # test_standardize_order_response('vs93qog7scpsoml9000iguk5')

    # .........................................................................
    # create ORDERS

    # test_buy_market(symbol=symbol, base_qty='15', quote_qty='10.00',
    #                 auto_borrow=False, base_or_quote='base')

    # sys.exit()

    # test_sell_market(symbol=symbol, base_qty='50', quote_qty='15.00',
    #                  auto_borrow=False, base_or_quote='base')

    # test_buy_limit(symbol=symbol, base_qty=base_qty)

    # print('='*80)
    # sleep(10)
    # test_create_stop_order(symbol=symbol, cancel=False)
    # print('='*80)
    # sleep(10)
    # test_cancel_all_orders()
    # print('='*80)
    # test_sell_all(asset=base_asset)


    # .........................................................................
    # CANCEL orders

    # test_cancel_order('vs8nqog6mdds119d000rt7tr', stop_order=True)

    # sleep(5)
    # test_cancel_all_orders(symbol=symbol)


    # .........................................................................
    # LOAN management
    # test_get_margin_config()
    # test_get_margin_risk_limit()
    # test_get_borrow_details_for_all()
    # test_get_borrow_details(runs=10)
    # test_get_borrow_details(asset='LaLa')
    # test_get_borrow_details(asset=symbol.split('-')[0])

    # test_get_liability(asset=asset)
    # print('-'*80)
    # print(' ')
    # test_repay(asset=asset)
    # test_repay_for_single_order(asset=base_asset)

    # .........................................................................
    # more COMPLEX TESTS

    # test_open_long_position(symbol=symbol, quote_qty=quote_qty)
    # test_open_short_position(symbol=symbol)
    # sleep(15)
    # print('creating stop market BUY order ...')
    # test_create_stop_order(symbol=symbol, side='BUY')

    # test_get_all_positions()

    # test_public_is_singleton()

    test_account_repository_response()

