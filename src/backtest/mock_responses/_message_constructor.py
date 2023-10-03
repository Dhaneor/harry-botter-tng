#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 129 23:50:04 2021

@author_ dhaneor
"""


from pandas.core import base
from broker.models.orders import MarketOrder
import string
import time
import json
from random import random, randrange, choice
from pprint import pprint
from copy import deepcopy
from helpers.timeops import unix_to_utc

from models.symbol import Symbol
from broker.models.orders import ExecutionReport
from exchange.binance_classic import Binance


# TODO inherit class from BaseSymbolConfig

# =============================================================================
class MockUSerStreamResponse:

    def __init__(self, symbol, commission_asset=None):

        self.symbol_name = symbol.symbol_name
        self.symbol = symbol
        
        self.commission = 0.00075
        if commission_asset is not None:
            self.commission_asset = commission_asset
        else: 
            self.commission_asset = self.symbol.quoteAsset

        self.slippage = 0.0015

        self.updated = 0
        self._update_prices()


    # -------------------------------------------------------------------------
    def new_order_message(self, order):

        if order.base_qty is not None: base_qty = order.base_qty
        else: base_qty = '0.00000000'
        
        if order.quote_qty is not None: quote_qty = order.quote_qty
        else: quote_qty = '0.00000000'
        
        if order.type == 'MARKET':  price = '0.00000000' 
        else: price = str(order.limit_price)

        if order.type == 'STOP_LIMIT' or order.type == 'STOP_MARKET':
            stop_price = order.stop_price
        else: stop_price = '0.00000000'

        order_id = self._get_random_id('order id')
        transact_time = round(time.time())
        event_time = transact_time + 1

        
        message = {"e": "executionReport",      # Event type
                    "E": event_time,            # Event time
                    "s": self.symbol_name,      # Symbol
                    "c": order.client_order_id, # Client order ID
                    "S": order.side,            # Side
                    "o": order.type,            # Order type
                    "f": "GTC",                 # Time in force
                    "q": base_qty,              # Order quantity
                    "p": price,                 # Order price
                    "P": stop_price,            # Stop price
                    "F": "0.00000000",          # Iceberg quantity
                    "g": -1,                    # OrderListId
                    "C": "",                    # Original client order ID; This is the ID of the order being canceled
                    "x": "NEW",                 # Current execution type
                    "X": "NEW",                 # Current order status
                    "r": "NONE",                # Order reject reason; will be an error code.
                    "i": order_id,              # Order ID
                    "l": "0.00000000",          # Last executed quantity
                    "z": "0.00000000",          # Cumulative filled quantity
                    "L": "0.00000000",          # Last executed price
                    "n": "0",                   # Commission amount
                    "N": 'null',                # Commission asset
                    "T": transact_time,         # Transaction time
                    "t": -1,                    # Trade ID
                    "I": 8641984,               # Ignore
                    "w": 'true',                # Is the order on the book?
                    "m": 'false',               # Is this trade the maker side?
                    "M": 'false',               # Ignore
                    "O": transact_time,         # Order creation time
                    "Z": "0.00000000",          # Cumulative quote asset transacted quantity
                    "Y": "0.00000000",          # Last quote asset transacted quantity (i.e. lastPrice * lastQty)
                    "Q": quote_qty              # Quote Order Qty
                    }
        
        return message


    def executed_order_message(self, order, no_of_fills=0):

        new_order_message = self.new_order_message(order)

        message = self.get_fill(new_order_message)[0]

        return message


    # -------------------------------------------------------------------------
    # returns a message or a number of messages for an order that was filled in
    # in one or more steps
    def get_fill(self, orig_message, number_of_fills=1):

        om = orig_message
        results = []

        side = om['S']
        tx_times = self._get_transaction_times(number_of_fills)
        prices = self._get_fill_prices(number_of_fills, side)
        quantities = self._get_quantities(float(om['q']), number_of_fills)
        cum_quote_asset = 0

        for idx in range(number_of_fills):

            results.append({})

            new = deepcopy(om)
            
            if (idx == number_of_fills -1): 
                new['X'] = 'FILLED'
            else: 
                new['X'] = 'PARTIALLY_FILLED'
            
            new['x'] = 'TRADE'
            new['E'] = tx_times[idx]
            new['T'] = tx_times[idx] -1
            new['l'] = quantities[idx]
            new['z'] = sum(quantities[:idx+1])
            new['L'] = prices[idx]
            new['Y'] = round(quantities[idx] * prices[idx], self.symbol.baseAssetPrecision)
            cum_quote_asset += round(quantities[idx] * prices[idx], self.symbol.baseAssetPrecision)
            new['Z'] = round(cum_quote_asset, self.symbol.quoteAssetPrecision)
            new['n'] = prices[idx]
            new['t'] = idx + 1

            if self.commission_asset == 'BNB':
                new['N'] = 'BNB'
                new['n'] = self._calculate_commission_bnb(om['S'], 
                                                             quantities[idx],
                                                             prices[idx]
                                                             )
            else:
                if side == 'BUY':
                    new['N'] = self.symbol.baseAsset
                    commission = self._calculate_buy_commission(quantities[idx])
                    new['n'] = commission

                elif side == 'SELL':
                    new['N'] = self.symbol.quoteAsset
                    commission = self._calculate_sell_commission(quantities[idx],
                                                               prices[idx]
                                                               )
                    print(commission)
                    new['n'] = commission

            results[idx] = new


        return results

    # -------------------------------------------------------------------------
    # helper functions 

    # get the latest prices for the symbol and for BNB - quote asset
    def _update_prices(self):

        pair = 'BNB' + self.symbol.quoteAsset
        with Binance() as conn: 
            self.bnb_price = conn.get_last_price(pair) 
            self.last_price = conn.get_last_price(self.symbol_name)
            self.updated = time.time()

    # create random id strings/numbers
    def _get_random_id(self, type):

        if type == 'client order id':

            let_num = string.ascii_letters
            let_num += string.digits
            length = 22

        elif type == 'trade id':

            let_num = string.digits
            length = 7

        elif type == 'order id':

            let_num = string.digits
            length = 8

        # build the result string
        res_list = [] 
        for idx in range(length):

            x = choice(let_num)
            res_list.append(x)

        return ''.join(res_list)

    def _get_transaction_times(self, number):

        tx_times = []
        for idx in range(number):

            time.sleep(0.1)
            tx_times.append(int(time.time()))

        return sorted(tx_times, reverse=False)

    def _get_fill_prices(self, number_of_fills, side):

        if (time.time() - self.updated > 300): self._update_prices()

        fill_prices = [self.last_price]
        factor = 1.2

        for idx in range(number_of_fills-1):

            if side == 'BUY':
                price = self.last_price * (1 + random() * self.slippage * factor) 
            elif side == 'SELL':
                price = self.last_price * (1 - random() * self.slippage * factor) 
            price = round(price, self.symbol.f_tickPrecision)
            fill_prices.append(price)

        if side == 'BUY': return sorted(fill_prices)
        else: return sorted(fill_prices, reverse=True)

    def _get_quantities(self, base_qty, number_of_fills):

        single_base_qty = base_qty / number_of_fills
        quantities = []
        precision = self.symbol.baseAssetPrecision

        for idx in range(number_of_fills-1):

            q = round(single_base_qty / 2.5 + (single_base_qty * random()), precision)
            quantities.append(q)

        quantities.append(base_qty - sum(quantities))
        return quantities

    def _calculate_buy_commission(self, base_qty):

        return base_qty * self.commission

    def _calculate_sell_commission(self, base_qty, price):

        return base_qty * price * self.commission

    def _calculate_commission_bnb(self, side, base_qty, price):
        
        com = (base_qty * price) * self.commission / self.bnb_price
        return round(com, 8)


# =============================================================================

if __name__ == '__main__':

    symbol_name = 'ADAUSDT'
    symbol = Symbol(symbol_name)

    order = MarketOrder(symbol=symbol,
                        market='SPOT',
                        side='SELL',
                        base_qty='50.00',
                        )

    erb = MockUSerStreamResponse(symbol, symbol.quoteAsset)
    no = erb.new_order_message(order)

    pprint(no)

    # -------------------------------------------------------------------------
    # for i in range(10): 
    #     test = erb._get_quantities(100, 5)

    #     print(test)
    #     print(sum(test))
    #     print('-'*25)

    # -------------------------------------------------------------------------
    fills = erb.get_fill(no, 3)

    # print(f'We got {len(fills)} fills\n')

    for fill in fills:

        er = ExecutionReport(fill)

        print('-=*=-'*10)
        human_time = unix_to_utc(fill['E']*1000)
        print(f'{human_time}\n')
        # pprint(fill)
        print(er)

    # -------------------------------------------------------------------------
    # cum_sl = 0
    # for i in range(20):
    #     p = erb._get_fill_prices(20, 'BUY')
    #     sl = round(p[i]/erb.last_price *100 - 100, 4)
    #     cum_sl += sl

    #     if i == 0: print(f'last price: {erb.last_price}')
    #     print(f'price: {p[i]} - slippage = {sl}%')
    
    # m = round(cum_sl / len(p), 4)
    # print(f'mean slippage: {m}%')