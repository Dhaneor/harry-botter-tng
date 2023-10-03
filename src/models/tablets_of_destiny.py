#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 15  18 15:55:23 2021

@author: dhaneor
"""
from distutils.command.build import build
import time
import string
import sys

from random import random, choice
from pprint import pprint
from abc import ABC, abstractmethod
from typing import List
from uuid import uuid1


from exchange.exchange import Exchange
from exchange.util.repositories import Repository
from broker.models.orders import Order
from broker.models.exchange_order import build_exchange_order
from models.symbol import Symbol
from helpers.timeops import seconds_to, unix_to_utc, time_difference_in_ms

# =============================================================================
class Position(ABC):

    def __init__(self, symbol:Symbol, repository:Repository):

        super().__init__()

        self.symbol = symbol
        self.repository = repository

        self.id = self._get_random_id(16)
        self.status = 'INITIALIZED'

        self.price_entry = 0
        self.value_entry = 0 

        self.price_exit = 0

        self._pnl_unrealized = 0 
        self._pnl_realized = 0 

        self.open_time = 0
        self.human_open_time = 0
        self.close_time = 0
        self.human_close_time = 0
        self.hold_time = 0

        self.order_ids: List[str] = []
        self.orders: List[Order] = []

        self.base_qty = 0
        self.quote_qty = 0
        self._value = 0

    def __repr__(self):

        out = '-=:*:=-' * 10
        out += f' [{self.symbol.symbol_name}] '
        if self.TYPE is not None: out += f'{self.TYPE} '
        out += f'position ({self.id}) {self.status} '
        out += '-=:*:=-' * 10

        return out

    @property
    def last_price(self):
        return float(
            [t for t in self.repository.tickers.tickers \
                if t['symbol'] == self.symbol.symbol_name
            ][0]['last'])
        
    @property
    def value(self):
        self._value = self._calculate_value()
        return 
    
    @property
    def unrealized_pnl(self):
        self._calculate_unrealized_pnl()
        return self._pnl_unrealized
        
    # -------------------------------------------------------------------------
    def include_order(self, order:Order):
        if self.status == 'INITIALIZED':
            self._open(order)
        else:
            self._add_order(order)
    
    # -------------------------------------------------------------------------
    # helper methods
    def _open(self, order:Order):
        
        if self._is_opening_order(order):
            self._add_order(order)
            
            er = order.last_execution_report
            message = er.original_message.get('message')
            
            self._set_open_time(message.get('transactTime'))
            
        else:
            raise Exception(
                f'[POSITION] Warning: {order.type} is invalid '
                f'for opening a position'
                )
        
        self.status = 'ACTIVE' if order.status == 'FILLED' else 'PENDING'
              
    def _close(self):
        pass
    
    def _add_order(self, order:Order):
        
        er = order.last_execution_report
        response = er.original_message.get('message')
        
        # make sure we have no duplicates and/or remove orders that 
        # were created (=limit order) and are now filled or partially
        # filled
        order_ids = [oid for oid in self.order_ids if oid != er.order_id]
        order_ids.append(er.order_id)
        self.order_ids = order_ids
        
        self.orders = [
            o for o in self.orders if o.order_id != response.get('order_id')
            ]

        self.orders.append(build_exchange_order(response))
        
        self.base_qty += float(response.get('executedQty'))
        self.quote_qty += float(
            response.get('cummulativeQuoteQty')
            )
        
        self.price_entry = round(
            self.quote_qty / self.base_qty, self.symbol.f_tickPrecision
        )
     
    @abstractmethod
    def _is_opening_order(self, order:Order) -> bool:
        pass

    @abstractmethod
    def _calculate_average_entry_price(self):
        pass
    
    @abstractmethod
    def _calculate_unrealized_pnl(self):
        pass
    
    # .......................................................................... 
    def _set_open_time(self, timestamp:int):
        self.open_time = timestamp
        self.human_open_time = unix_to_utc(self.open_time)
            
    def _calculate_value(self):
        print(type(self.base_qty), type(self.last_price))
        return round(
            self.base_qty * self.last_price, self.symbol.quoteAssetPrecision
            )
   
    def _calculate_hold_time(self):
        self.hold_time = round(time_difference_in_ms(self.open_time) / 1000)

    def _get_random_id(self, length:int) -> str:
        return str(uuid1())
        return ''.join(
            [choice(string.digits) for _ in range(length)]
            )


class LongPosition(Position):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.TYPE = 'LONG'
        
    # --------------------------------------------------------------------------
    def _is_opening_order(self, order:Order) -> bool:
        return True if not 'STOP' in order.type else False
    
    def _calculate_average_entry_price(self):
        
        self.price_entry = round(self.quote_qty / self.base_qty,
                                 self.symbol.f_tickPrecision)
    
    def _calculate_unrealized_pnl(self):
        
        self._pnl_unrealized = round(self.value - self.quote_qty,
                                     self.symbol.f_tickPrecision)
    
    








# =============================================================================
class Account:
    """Class that acts as a proxy for the account on the exchange.
    
    This is like a central repository for things and informations that
    relate to our account in the exchange. Here we have all the 
    balances for the coins that are tradeable in the selected market.
    It also provides access to informations that are related to loan 
    management, e.g. the max borrow amount for any one coin/asset.
    
    It uses some specialized repository classes/objects that cache 
    important information. This reduces the amount of necessary API
    calls down to a minmum and dramatically improves speed if we 
    need the informations multiple times. 
    """
    def __init__(self, exchange:Exchange):
        """Initializes the account.

        :param exchange: _description_, defaults to 'kucoin'
        :type exchange: str, optional
        :param market: _description_, defaults to 'CROSS MARGIN'
        :type market: str, optional
        """
        self.exchange = exchange
        self._assets : dict = {}
        
    @property
    def account(self) -> list:
        return self.exchange.account
    
    @property    
    def assets_that_we_own(self) -> list:
        return [a['asset'] for a in self.account if a['net'] != 0]
        
    @property 
    def tickers(self) -> list:
        return self.exchange.get_all_tickers()
    
    @property
    def margin_info(self) -> list:
        return self.exchange.margin_loan_info
    
    # --------------------------------------------------------------------------
    def get_account_value(self, quote_asset:str=None):
        """Calculates the account value in the given quote asset currency.

        :param quote_asset: which asset should be used as quote asset
        for the calculation, defaults to None
        :type quote_asset: str, optional
        :return: the calculated value (default: in USDT)
        :rtype: float
        """
        acc = self.account 
        acc = [bal for bal in acc if bal['net'] != 0]
        acc = [self._add_value_to_balance(bal) for bal in acc]
        
        if quote_asset is None or quote_asset == 'USDT':
            return sum([bal['value']['USDT'] for bal in acc])
        elif quote_asset == 'BTC':
            return sum([bal['value']['BTC'] for bal in acc])
        else:
            return None
    
    def get_asset(self, asset:str) -> dict:
        
        _asset = list(filter(lambda x: x['asset'] == asset, self.account))
        if _asset:
            margin_info = list(filter(
                lambda x: x['asset'] == asset, self.margin_info))
            
            if margin_info:
                return {**_asset[0], **margin_info[0]} 
        
        return None
          
    # --------------------------------------------------------------------------    
    def _add_value_to_balance(self, balance:dict) -> dict:
        
        balance['value'] = {}
        asset = balance['asset']
        symbol = f'{asset}-USDT'
        balance['symbol'] = symbol
        btc_price = 0
         
        for ticker in self.tickers:
            
            if ticker['symbol'] == 'BTC-USDT':
                btc_price = float(ticker['last'])
            
        for ticker in self.tickers: 
            if asset == 'USDT':
                balance['value']['USDT'] = balance['net']
                 
              
            if ticker['symbol'] == symbol:
                last_price = float(ticker['last'])
                net_amount = balance['net']
                balance['value']['USDT'] = round(
                    last_price * net_amount, 8) 
                
                break

        balance['value']['BTC'] = round(
            balance['value']['USDT'] / btc_price, 8)        
        return balance
            
                

# =============================================================================

if __name__ == '__main__':

    pass
    

































# =============================================================================

'''
    message = {'C': '',
                'E': 1618268329645, 
                'F': '0.00000000', 
                'I': 2651735496, 
                'L': '1.33387000', 
                'M': True, 
                'N': 'BNB',
                'O': 1618268329644,
                'P': '0.00000000',
                'Q': '0.00000000',
                'S': 'SELL',
                'T': 1618268329644,
                'X': 'FILLED',
                'Y': '20.00805000',
                'Z': '20.00805000',
                'c': 'web_0b1087f19b254ba288fb7e105a651c74',
                'e': 'executionReport',
                'f': 'GTC',
                'g': -1,
                'i': 1269086272,
                'l': '15.00000000',
                'm': False,
                'n': '0.00002484',
                'o': 'MARKET',
                'p': '0.00000000',
                'q': '15.00000000',
                'r': 'NONE',
                's': 'ADAUSDT',
                't': 122083746,
                'w': False,
                'x': 'TRADE',
                'z': '15.00000000'}
'''


























'''

format of binance response for filled message from USER STREAM:

1) actual market message filled

{'C': '',
 'E': 1618268329645, 
 'F': '0.00000000', 
 'I': 2651735496, 
 'L': '1.33387000', 
 'M': True, 
 'N': 'BNB',
 'O': 1618268329644,
 'P': '0.00000000',
 'Q': '0.00000000',
 'S': 'SELL',
 'T': 1618268329644,
 'X': 'FILLED',
 'Y': '20.00805000',
 'Z': '20.00805000',
 'c': 'web_0b1087f19b254ba288fb7e105a651c74',
 'e': 'executionReport',
 'f': 'GTC',
 'g': -1,
 'i': 1269086272,
 'l': '15.00000000',
 'm': False,
 'n': '0.00002484',
 'o': 'MARKET',
 'p': '0.00000000',
 'q': '15.00000000',
 'r': 'NONE',
 's': 'ADAUSDT',
 't': 122083746,
 'w': False,
 'x': 'TRADE',
 'z': '15.00000000'}

 {'clientOrderId': '0z2OBEqcsnrLwosx9pNxV2',
 'cumulativeQuoteQty': '122.08708534',
 'executedQty': '100',
 'fills': [{'commission': 0.0204,
            'commissionAsset': 'ADA',
            'price': '1.22109548',
            'qty': '20.4',
            'tradeId': '9028182'},
           {'commission': '5.496e-05',
            'commissionAsset': 'BNB',
            'price': '1.22143013',
            'qty': '15.0',
            'tradeId': '9028182'},
           {'commission': '1.979e-05',
            'commissionAsset': 'BNB',
            'price': '1.22161531',
            'qty': '5.4',
            'tradeId': '9028182'},
           {'commission': 0.0592,
            'commissionAsset': 'ADA',
            'price': '1.22232031',
            'qty': '59.2',
            'tradeId': '9028182'}],
 'messageId': '77218711',
 'messageListId': -1,
 'origQty': '100',
 'price': '0.00000000',
 'side': 'BUY',
 'status': 'FILLED',
 'symbol': 'ADAUSDT',
 'timeInForce': 'GTC',
 'transactTime': 1618273515.9356642,
 'type': 'MARKET'}


2) newly created market order

{'C': '',
 'E': 1618268317668,
 'F': '0.00000000',
 'I': 2651732693,
 'L': '0.00000000',
 'M': False,
 'N': None,
 'O': 1618268317667,
 'P': '0.00000000',
 'Q': '20.13252800',
 'S': 'BUY',
 'T': 1618268317667,
 'X': 'NEW',
 'Y': '0.00000000',
 'Z': '0.00000000',
 'c': 'web_e29a71cc70ea4ed4b5d81e3e18780582',
 'e': 'executionReport',
 'f': 'GTC',
 'g': -1,
 'i': 1269084935,
 'l': '0.00000000',
 'm': False,
 'n': '0',
 'o': 'MARKET',
 'p': '0.00000000',
 'q': '15.00000000',
 'r': 'NONE',
 's': 'ADAUSDT',
 't': -1,
 'w': True,
 'x': 'NEW',
 'z': '0.00000000'}

3) example from api documentation

{
  "e": "executionReport",        // Event type
  "E": 1499405658658,            // Event time
  "s": "ETHBTC",                 // Symbol
  "c": "mUvoqJxFIILMdfAW5iGSOW", // Client order ID
  "S": "BUY",                    // Side
  "o": "LIMIT",                  // Order type
  "f": "GTC",                    // Time in force
  "q": "1.00000000",             // Order quantity
  "p": "0.10264410",             // Order price
  "P": "0.00000000",             // Stop price
  "F": "0.00000000",             // Iceberg quantity
  "g": -1,                       // OrderListId
  "C": "",                       // Original client order ID; This is the ID of the order being canceled
  "x": "NEW",                    // Current execution type
  "X": "NEW",                    // Current order status
  "r": "NONE",                   // Order reject reason; will be an error code.
  "i": 4293153,                  // Order ID
  "l": "0.00000000",             // Last executed quantity
  "z": "0.00000000",             // Cumulative filled quantity
  "L": "0.00000000",             // Last executed price
  "n": "0",                      // Commission amount
  "N": null,                     // Commission asset
  "T": 1499405658657,            // Transaction time
  "t": -1,                       // Trade ID
  "I": 8641984,                  // Ignore
  "w": true,                     // Is the order on the book?
  "m": false,                    // Is this trade the maker side?
  "M": false,                    // Ignore
  "O": 1499405658657,            // Order creation time
  "Z": "0.00000000",             // Cumulative quote asset transacted quantity
  "Y": "0.00000000",             // Last quote asset transacted quantity (i.e. lastPrice * lastQty)
  "Q": "0.00000000"              // Quote Order Qty
}

'''

