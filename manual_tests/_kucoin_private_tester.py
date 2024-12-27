#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import cProfile

from time import sleep
from pprint import pprint
from typing import Union, Optional, List
from uuid import uuid1

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

from src.exchange.kucoin_ import KucoinFactory, KucoinCrossMargin
from src.broker.models.symbol import Symbol, SymbolFactory
from src.broker.models.exchange_order import build_exchange_order
from util.timeops import seconds_to, unix_to_utc, execution_time

from broker.config import CREDENTIALS

factory = KucoinFactory()
conn = factory.build_client(market='cross margin', credentials=CREDENTIALS)

sf = SymbolFactory(exchange='kucoin', market='cross margin')

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

def _func():
    conn.get_orders()

def profile_get_orders():
    cProfile.run('_func()', sort='tottime')

# -----------------------------------------------------------------------------
# ACCOUNT related
@execution_time
def test_get_accounts(currency: Optional[str]=None):
    res = conn.account.get_accounts(currency=currency, account_type='margin')

    if res['success']:
        accounts = res['message']
        acc_non_zero = [cur for cur in accounts if not cur['balance']=='0']
        pprint(acc_non_zero)
        print('-'*80)
    else:
        pprint(res)

@execution_time
def test_get_account_by_id(id_:str):
    res = conn.account.get_account_by_id(id_)

    if res['success']:
        pprint(res['message'])
        print('-'*80)
    else:
        pprint(res)

@execution_time
def test_get_withdrawal_quota(currency:str='BTC'):
    res = conn.get_withdrawal_quotas(currency=currency)

    if res['success']: pprint(res['message'])
    else: pprint(res)

@execution_time
def test_get_account(interesting:list=None, non_zero:bool=False):
    accounts = conn.get_account()

    if not accounts:
        return

    if interesting:
        accounts = [acc for acc in accounts if acc['asset'] in interesting]

    if non_zero:
        accounts = [acc for acc in accounts if float(acc['total']) != 0]
    pprint(accounts)

@execution_time
def test_get_balance(asset:str):
    res = conn.get_balance(asset=asset)

    if res['success']:
        balance = res['message']
        pprint(balance)
    else:
        balance = {}
        pprint(res)

    return balance

@execution_time
def test_get_fees(symbols: Optional[List[str]]=None):
    res = conn.get_fees(symbols=symbols)

    if res['success']:
        fees = res['message']
        pprint(fees)
    else:
        pprint(res)


# .............................................................................
# get ORDERs
@execution_time
def test_get_orders(symbol=None, side: Optional[str]=None,
                    order_type: Optional[str]=None,
                    start=None, end=None):

    res = conn.get_orders( #type:ignore
        symbol=symbol, side=side, order_type=order_type,
        start=start, end=end
    )

    if res['success']:
        orders = res['message']
        [pprint(build_exchange_order(order)) for order in orders]
        print(f'fetched {len(orders)} orders')
    else:
        pprint(res)

@execution_time
def test_get_multiple_orders(symbol: Optional[str]=None, start=None, end=None):
    res = conn._get_multiple_orders(
        symbol=symbol, side='SELL' ,start=start, end=end
    )

    if res.get('success'):
        orders = res['message']
        [print_order(order) for order in orders]
        print(f'fetched {len(orders)} orders')
    else:
        pprint(res)

@execution_time
def test_get_active_orders(symbol: Optional[str]=None, side: Optional[str]=None):
    orders = conn.get_active_orders(symbol=symbol, side=side)

    if not orders:
        print(f'There are no active orders for {symbol} ({side=})')
        return

    [print_order(order) for order in orders]
    print(f'fetched {len(orders)} orders')

@execution_time
def test_get_active_stop_orders(symbol:str=None, side=None):
    res = conn.get_active_stop_orders(symbol=symbol)

    if res.get('success'):
        orders = res['message']
        [print_order(order) for order in orders]
        print(f'fetched {len(orders)} orders')
    else:
        pprint(res)

@execution_time
def test_get_order(order_id:str=None, client_order_id:str=None):
    res = conn.get_order(
        order_id=order_id, client_order_id=client_order_id
    )

    pprint(res)

@execution_time
def test_get_fills(order_id:str=None, start:str=None):
    res = conn.get_fills(order_id=order_id, start=start)

    if res['success']:
        pprint(res['message'])
    else:
        pprint(res)


# .............................................................................
# create ORDERS
@execution_time
def test_buy_market(symbol=None, base_qty=None, quote_qty=None,
                    base_or_quote='quote', auto_borrow=False) -> dict:
    client_order_id = str(uuid1())

    if base_or_quote == 'base':
        res = conn.buy_market(
            symbol=symbol, client_order_id=client_order_id, base_qty=base_qty,
            auto_borrow=auto_borrow
        )
    else:
        res = conn.buy_market(
            symbol=symbol, client_order_id=client_order_id, quote_qty=quote_qty,
            auto_borrow=auto_borrow
        )

    pprint(res['message'])
    return res

@execution_time
def test_sell_market(symbol=None, base_qty=None, quote_qty=None,
                     base_or_quote='base', auto_borrow=False):
    client_order_id = str(uuid1())

    if base_or_quote == 'base':
        res = conn.sell_market(
            symbol=symbol, client_order_id=client_order_id, base_qty=base_qty,
            auto_borrow=auto_borrow
        )
    else:
        res = conn.sell_market(
            symbol=symbol, client_order_id=client_order_id, quote_qty=quote_qty,
            auto_borrow=auto_borrow
        )

    pprint(res['message'])
    return res

@execution_time
def test_buy_limit(symbol=None, base_qty=None, price=None, auto_borrow=False):

    with EXCHANGE() as conn:

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
    with EXCHANGE() as conn:
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
    s = None
    res = conn.public.get_symbol(symbol)
    if res['success']:
        s = sf.build_symbol_from_api_response(res['message'])

    if not s:
        print(f'unable to get symbol information ... quitting')
        return

    price_precision = s.tick_precision
    base_step, base_precision = s.lot_size_step_size,s.lot_size_step_precision
    last_price, balance_free = None, None

    res = conn.public.get_ticker(symbol)
    if res['success']:
        last_price = float(res['message']['price'])
    else:
        pprint(res)
        return

    # get the current balance for the base asset
    balance = test_get_balance(asset=s.base_asset)
    pprint(balance)
    balance_free = float(balance.get('free', 0))
    balance_borrowed = float(balance.get('borrowed', 0))

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

    print('-'*80)
    if res['success']:
        # order = res['message']
        pprint((order := res['message']))

        if cancel:
            sleep(1)
            res = conn.cancel_order(str(order['orderId']))
            print('-'*80)
            pprint(res)

        test_get_active_stop_orders(symbol=symbol)
        test_get_balance(asset=base_asset)
    else:
        pprint(res)

# .............................................................................
# cancel ORDERS
@execution_time
def test_cancel_order(order_id:str, stop_order:bool):
    res = conn.cancel_order(order_id, stop_order=stop_order)

    pprint(res)

@execution_time
def test_cancel_all_orders(symbol=None):
    res = conn.cancel_all_orders(symbol=symbol)

    if res['success']:
        print(res['message'])
        cancelled = res['message']['cancelledOrderIds']
        number_deleted = len(cancelled)
        [pprint(item) for item in cancelled]
        print('-'*80)
        print(f'{number_deleted} orders cancelled ...')
    else:
        pprint(res)


# .............................................................................
# LOAN management
@execution_time
def test_get_margin_config():
    res = conn.get_margin_config()

    if res['success']:
        pprint(res.get('message'))

@execution_time
def test_get_margin_risk_limit():
    res = conn.get_margin_risk_limit()

    if res.get('success'):
        [print(i) for i in res.get('message')]
    else:
        pprint(res)

@execution_time
def test_get_borrow_details_for_all():
    res = conn.get_borrow_details_for_all()

    if res.get('success'):
        [print(i) for i in res.get('message')]
    else:
        pprint(res)

@execution_time
def test_get_borrow_details(asset:str):
    res = conn.get_borrow_details(asset=asset)

    if res['success']:
        message = res.get('message')
        pprint(message)
        return message
    else:
        pprint(res)
        return None

@execution_time
def test_get_liability(asset:str=None):
    res = conn.get_liability(asset=asset)

    if res['success']:
        pprint(res['message'])
    else:
        pprint(res)

@execution_time
def test_borrow(asset: str, size: Optional[float]=None):
    if not size:
        borrow_details = test_get_borrow_details(asset)
        if not borrow_details:
            return
        size = borrow_details['maxBorrowSize'] / 2

    print(asset, size)

    res = conn.borrow(currency=asset, size=size)
    if res['success']:
        test_get_borrow_details(asset)
    else:
        pprint(res)

@execution_time
def test_repay(asset:str=None):
    res = conn.get_liability(asset=asset)

    if res['success']:
        orders = res['message']

        if orders:

            liability = sum([float(o['liability']) for o in orders])

            res = conn.get_balance(asset=asset)

            if res['success']:
                available = float(res['message']['free'])
            else: available = 0

            size = min(available, liability)
            print(f'going to repay {size} {asset} ...')

            if size > 0:
                res = conn.repay(currency=asset,
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

    with EXCHANGE() as conn:
        res = conn._standardize_fill_response([fill, fill, fill])

        pprint(res)

@execution_time
def test_standardize_order_response(order_id:str):
    with EXCHANGE() as conn:
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

    with EXCHANGE() as conn:
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

    with EXCHANGE() as conn:
        res = conn.sell_market(symbol=symbol, client_order_id=client_order_id,
                                base_qty=pos_size, auto_borrow=True)
    if res.get('success'):
        message = res.get('message')
        pprint(message)
    else:
        pprint(res)
        sys.exit()

    with EXCHANGE() as conn:
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


    res = conn.public.get_symbols(quote_asset='USDT')
    message = res.get('message') if res.get('success') else []
    symbols = [item['symbol'] for item in message]
    tokens = [item.split('-')[0] for item in symbols]

    res = conn.get_account()
    account = res.get('message') if res.get('success') else []
    account = [
        item for item in account if item.get('asset') in tokens\
        and any(
            (float(item.get('total')) != 0, float(item.get('borrowed')) != 0)
        )
    ]

    [print(i) for i in account]

    tokens = [item.get('asset') for item in account]

    print(tokens)

    res = conn.public.get_all_tickers()
    tickers= res.get('message') if res.get('success') else []
    tickers = [
        item for item in tickers \
            if item.get('symbol').split('-')[0] in tokens
    ]

    print(f'found {len(tickers)} tickers')
    [print(t['symbol'], t['last']) for t in sorted(tickers, key=lambda x: x['symbol'])]

@execution_time
def test_get_filled_orders(symbol:str):
    with EXCHANGE() as conn:
        res = conn.get_orders(symbol=symbol, status='FILLED')

    if res['success']:
        orders = res['message']
        filled = [o for o in orders if o['status'] == 'FILLED']
        for o in filled:
            print('-'*80)
            print(unix_to_utc(o['updateTime']))
            pprint(o)


@execution_time
def test_repay_for_single_order(asset:str=None):
    with EXCHANGE() as conn:
        res = conn.get_liability(asset=asset)

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

        res = conn.get_balance(asset=asset)
        if res['success']:
            pprint(res['message'])

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
        binance_symbol = None

    if isinstance(conn, KucoinCrossMargin):
        symbol = kucoin_symbol
    else:
        symbol = binance_symbol

    interval = '12h'
    start = None # 'November 02, 2022 00:00:00'
    end = None #'November 30, 2022 00:00:00'

    base_qty = 10.00
    quote_qty = 5.00

    symbols = ['BTC-USDT', 'XRP-USDT', 'ETH-BTC']

    # .........................................................................
    # ACCOUNT related methods
    #
    # test_get_accounts()
    # test_get_withdrawal_quota()
    # test_get_account(interesting=['XLM', 'USDT'])
    test_get_account(non_zero=False)
    # test_get_balance(asset=quote_asset)
    # test_get_balance(asset=base_asset)
    # test_get_fees(symbols=symbols)


    # .........................................................................
    # get ORDERS
    #
    # test_get_orders(symbol=symbol, order_type=None, side=None,
    #                 start=start, end=end)
    # profile_get_orders()

    # print('-=*=-'*15)

    # test_get_multiple_orders(symbol=symbol, start=start, end=end)
    # test_get_active_orders(symbol=symbol, side=None)
    # test_get_active_stop_orders(symbol=None, side=None)

    # BINANCE orders
    # test_get_order(symbol=symbol, order_id='4037761293')

    # KUCOIN orders
    # test_get_order(symbol=symbol, order_id='620848e0fd31e50001a59db2') # MARKET BUY FILLED
    # test_get_order(symbol=symbol, order_id='620782247ee3b40001c73613') # LIMIT BUY FILLED

    # test_get_order(client_order_id='6206e1f4f235a400017ef74b')

    # test_get_fills(order_id='620782247ee3b40001c73613')
    # test_get_fills(start='1644489741717')

    # test_get_filled_orders(symbol=symbol)
    # test_get_fills(order_id='620782247ee3b40001c73613')

    # test_fill_transformation()
    # test_standardize_order_response('620782247ee3b40001c73613')
    # test_standardize_order_response('vs93qog7scpsoml9000iguk5')

    # .........................................................................
    # create ORDERS

    # test_buy_market(symbol=symbol, base_qty=base_qty, quote_qty=quote_qty,
    #                 auto_borrow=False, base_or_quote='base')

    # sys.exit()

    # test_sell_market(symbol=symbol, base_qty=base_qty, quote_qty=quote_qty,
    #                  auto_borrow=False, base_or_quote='base')

    # test_buy_limit(symbol=symbol, base_qty=base_qty)

    # print('='*80)
    # sleep(10)

    # for _ in range(5):
    #     test_create_stop_order(symbol=symbol, cancel=False)
    # print('='*80)
    # sleep(10)
    # test_cancel_all_orders(symbol=symbol)
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
    # test_get_borrow_details(asset=symbol.split('-')[1])
    # test_get_borrow_details(asset='LaLa')
    # test_get_borrow_details(asset=symbol.split('-')[0])

    # test_get_liability(asset=None)
    # test_borrow(asset=quote_asset)
    # print('-'*80)
    # print(' ')
    # sleep(3)
    # test_repay(asset=quote_asset)
    # test_repay_for_single_order(asset=base_asset)

    # .........................................................................
    # more COMPLEX TESTS

    # test_open_long_position(symbol=symbol, quote_qty=quote_qty)
    # test_open_short_position(symbol=symbol)
    # sleep(15)
    # print('creating stop market BUY order ...')
    # test_create_stop_order(symbol=symbol, side='BUY')

    # test_get_all_positions()

