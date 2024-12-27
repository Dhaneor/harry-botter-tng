#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 02 13:47:38 2021

@author: dhaneor
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 129 23:50:04 2021

@author_ dhaneor
"""
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

import string
import time
from random import random, randrange, choice

from broker.models.symbol import Symbol
from broker.models.orders import (
    Order, MarketOrder, StopMarketOrder, LimitOrder, 
    StopLimitOrder, CancelOrder
)


# =============================================================================
class OrderResultConstructor:

    def __init__(self, symbol:object):

        self.symbol: object = symbol
        self.max_slippage: float = 0.0015
        self.actual_slippage: float = 0.00

        self.BASE_ASSET = ''
        self.QUOTE_ASSET = ''
        self.COMMISSION = 0.001
        self.COMMISSION_BNB = 0.00075

        self.BASE_ASSET_PRECISION = 0
        self.QUOTE_ASSET_PRECISION = 0

        self.BASE_COMMISSION_PRECISION = 0 
        self.QUOTE_COMMISSION_PRECISION = 0 
        
        self.BNB_COMMISSION_PRECISION = 8
        self.BNB_PRICE = 250

        self.TICK_SIZE = 0
        self.STEP_SIZE = 0 
        self.STEP_PRECISION = 0
        self.TICK_PRECISION = 0

        self.order_type = None
        self.error_message = f'This needs to be implemented for {self.order_type}!'

    # -------------------------------------------------------------------------
    # construct and return the result as a structured response like it will be
    # returned by the exchange client class. this should be the main entry point
    # for all kinds of orders. 
    #
    # keyword arguments:
    #
    # order :: order object (MarketOrder, LimitOrder, StopLossMarketOrder, StopLossLimitOrder)
    # status_code :: int (for one of the Binance/exchange status codes) 
    #                or str = 'random' for a randomly selected error code
    #                -> status code for successful operation is 200 
    def execute(self, order:Order) -> dict:
        self.order_type = order.type
        
        try:
            if isinstance(order, MarketOrder):
                order_result = self._get_market_order_result(order=order)
                res = self._build_happy_return(order_result)

            if isinstance(order, LimitOrder):
                order_result = self._get_limit_order_result(order=order)
                res = self._build_happy_return(order_result)

            if isinstance(order, StopMarketOrder):
                self.order_type = 'Limit Order'
                raise NotImplementedError

            if isinstance(order, StopLimitOrder):
                order_result = self._get_stop_limit_order_result(order=order)
                res = self._build_happy_return(order_result)

            if isinstance(order, CancelOrder):
                self.order_type = 'Limit Order'
                raise NotImplementedError

        except NotImplementedError:
            self.error_message = 'You tried to call a method that is not implemented yet!'
            res = self._build_unhappy_return(status_code=-1)
            
        except Exception as e:
            print(e)
            res = self._build_unhappy_return(status_code=-1)
        
        finally: 
            return res

        
    # -------------------------------------------------------------------------
    # construct and return the result for a market BUY or SELL order
    def _get_market_order_result(self, order) -> dict:

        self._initialize_symbol(order.symbol)
        
        if order.client_order_id is None: 
            client_order_id = self._get_random_id('client order id')
        else: 
            client_order_id = order.client_order_id
            
        fills = self._get_fills(side=order.side, 
                                base_qty=order.base_qty, 
                                base_price=order.last_price)

        price, executed_base_qty, executed_quote_qty = self._calculate_fill_price(fills)
        
        order_id = self._get_random_id('order id')
        orig_qty = str(order.base_qty)
        transact_time = int(time.time())
        
        return {'clientOrderId': client_order_id,
                'cummulativeQuoteQty': str(executed_quote_qty),
                'executedQty': str(executed_base_qty),
                'fills': fills,
                'orderId': order_id,
                'orderListId': -1,
                'origQty': orig_qty,
                'price': str(price),
                'side': order.side,
                'status': 'FILLED',
                'symbol': order.symbol.symbol_name,
                'timeInForce': 'GTC',
                'transactTime': transact_time,
                'time': transact_time,
                'type': 'MARKET'}


    # -------------------------------------------------------------------------
    # construct and return the result for a LIMIT BUY or SELL order
    def _get_limit_order_result(self, order) -> dict:

        self._initialize_symbol(order.symbol)
        
        if order.client_order_id is None: 
            client_order_id = self._get_random_id('client order id')
        else: 
            client_order_id = order.client_order_id
            
        fills = self._get_fills(side=order.side, 
                                base_qty=order.base_qty, 
                                base_price=order.last_price)
        
        price, executed_base_qty, executed_quote_qty = self._calculate_fill_price(fills)
        
        order_id = self._get_random_id('order id')
        orig_qty = str(order.base_qty)
        transact_time = time.time() * 1000
        
        return {'clientOrderId': client_order_id,
                'cummulativeQuoteQty': str(executed_quote_qty),
                'executedQty': str(executed_base_qty),
                'fills': fills,
                'orderId': order_id,
                'orderListId': -1,
                'origQty': orig_qty,
                'price': str(price),
                'side': order.side,
                'status': 'FILLED',
                'symbol': order.symbol,
                'timeInForce': 'GTC',
                'transactTime': int(transact_time),
                'type': 'MARKET'}


    # -------------------------------------------------------------------------
    # construct and return the result for a STOP LOSS LIMIT order
    def _get_stop_limit_order_result(self, order:StopLimitOrder, 
                                     filled:bool=False) -> dict:

        symbol = order.symbol
        self._initialize_symbol(symbol)

        side = order.side
        base_qty = order.base_qty
        stop_price, limit_price = order.stop_price, order.limit_price

        # if parameter 'filled' was set to True, return a resutl that reflects
        # a SL order that was triggered/filled
        if filled:

            fills = self._get_fills(side=side, base_qty=base_qty, base_price=stop_price)
            price, base_qty, quote_qty = self._calculate_fill_price(fills)             
            transact_time = time.time()

            # calculate slippage (just for reference and sanity check)
            self.actual_slippage = (price / stop_price - 1) * 100 
            
            return {'clientOrderId': order.client_order_id,
                    'cummulativeQuoteQty': str(quote_qty),
                    'executedQty': str(base_qty),
                    'fills': fills,
                    'orderId': str(self._get_random_id('order id')),
                    'orderListId': -1,
                    'origQty': str(order.base_qty),
                    'price': str(price),
                    'side': side,
                    'status': 'FILLED',
                    'symbol': self.symbol,
                    'timeInForce': 'GTC',
                    'transactTime': int(transact_time),
                    'type': 'STOP_LOSS_LIMIT'}

        # otherwise return a result for a SL order that was just creeted
        else:
            return {'clientOrderId': order.client_order_id,
                    'cummulativeQuoteQty': '0.00000000',
                    'executedQty': '0.00000000',
                    'fills': [],
                    'orderId': str(self._get_random_id('order id')),
                    'orderListId': -1,
                    'origQty': str(base_qty),
                    'price': str(limit_price),
                    'side': side,
                    'status': 'NEW',
                    'stopPrice': str(stop_price),
                    'symbol': self.symbol,
                    'timeInForce': 'GTC',
                    'transactTime': int(time.time()-50),
                    'type': 'STOP_LOSS_LIMIT'
                    }



    # -------------------------------------------------------------------------
    # this builds up and then reduces a complete long position from start to
    # finish for testing purposes
    def get_complete_long_position(self, symbol, base_price, cum_base_qty, count=1):

        self.symbol = symbol
        self._initialize_symbol(symbol)
        
        amount = cum_base_qty / count

        # calculate random transaction times and create two lists 
        # one for 'BUYS' and one for 'SELLS
        tx_times = []
        for idx  in range(count*2):

            tx_times.append(int(time.time() - random()**2 * 100000))

        tx_times = sorted(tx_times, reverse=False) 
        tx_times_dict = {'BUY' : tx_times[:count],
                         'SELL' : tx_times[count:]
                         }

        # build/calculate the values that we need for the construction
        # of the dict which has the same structure as the one returned 
        # from binance when an order gets filled 
        for i in range(2):
            
            side = 'BUY' if i == 0 else 'SELL'
            if side == 'BUY': tx_times = tx_times_dict.get('BUY') 
            else: tx_times = tx_times_dict.get('SELL')
            
            for idx in range(count):

                
                price = base_price + (random() * base_price) / 10 

                client_order_id = self._get_random_id('client order id')
                quote_qty = round(amount * price, self.QUOTE_ASSET_PRECISION)
                executed_qty = str(round(amount, self.BASE_ASSET_PRECISION))
                fills = self._get_fills(side, amount, quote_qty)
                order_id = self._get_random_id('order id')
                order_list_id = -1
                orig_qty = str(round(amount, self.BASE_ASSET_PRECISION))
                price = '0.00000000'
                transact_time = tx_times[idx]
                
                res = {'clientOrderId': client_order_id,
                        'cumulativeQuoteQty': str(quote_qty),
                        'executedQty': executed_qty,
                        'fills': fills,
                        'orderId': order_id,
                        'orderListId': order_list_id,
                        'origQty': orig_qty,
                        'price': price,
                        'side': side,
                        'status': 'FILLED',
                        'symbol': self.symbol,
                        'timeInForce': 'GTC',
                        'transactTime': transact_time,
                        'type': 'MARKET'}

                self.trades.append(res)

                # print(f'Slippage = {round(self.slippage, 4)} %')

        return self.trades


    # --------------------------------------------------------------------------------
    # helper functions 

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

    # create some simulated fills 
    def _get_fills(self, side:str, base_qty:float, base_price:float) -> list:

        part_qtys = self._make_up_fill_quantitities(base_qty=base_qty)
        number_of_fills = len(part_qtys)
        
        if self.order_type == 'MARKET' or 'STOP' in self.order_type:
            prices = self._get_randomized_prices(base_price=base_price, 
                                                 count=number_of_fills)
        else:
            prices = [base_price] * number_of_fills
        
        # sort the prices differently for 'BUYS' and 'SELLS'
        if side == 'BUY': prices = sorted(prices)
        else: prices = sorted(prices, reverse=True)

        # calculate commision(s)
        commissions, commission_assets = [], []
        
        for idx in range(number_of_fills):

            commission_from_bnb = False

            # calculate the commission
            if side == 'BUY' and not commission_from_bnb:
                commission = part_qtys[idx] * self.COMMISSION
                commissions.append((round(commission, self.BASE_COMMISSION_PRECISION)))
                commission_assets.append(self.BASE_ASSET)
                
            elif side == 'BUY' and commission_from_bnb:
                commission = part_qtys[idx] * prices[idx] / self.BNB_PRICE * self.COMMISSION_BNB
                commissions.append(str(round(commission, self.BNB_COMMISSION_PRECISION)))
                commission_assets.append('BNB')

            elif side == 'SELL' and not commission_from_bnb:
                commission = part_qtys[idx] * prices[idx] * self.COMMISSION
                commissions.append(str(round(commission, self.QUOTE_COMMISSION_PRECISION)))
                commission_assets.append(self.QUOTE_ASSET)
                
            else:
                commission = part_qtys[idx] * prices[idx] / self.BNB_PRICE * self.COMMISSION_BNB
                commissions.append(str(round(commission, self.BNB_COMMISSION_PRECISION)))
                commission_assets.append('BNB')
            
            # round and stringify the prices
            prices[idx] = str(round(prices[idx], self.QUOTE_ASSET_PRECISION))
            

        # build the dictionary and append to the 'fills' list
        fills = []

        for idx in range(number_of_fills):
            fills.append({'commission' : commissions[idx],
                          'commissionAsset' : commission_assets[idx],
                          'price' : prices[idx],
                          'qty' : str(part_qtys[idx]),
                          'tradeId' : self._get_random_id('trade id')})        
        return fills

    def _make_up_fill_quantitities(self, base_qty:float) -> list:

        res = []
        # randomize the number of fills
        count = 2 + round(random() * 5)
        
        # calculate partial quantities and prices
        # set the last quantity to a negative amount beforehand and repaet the 
        # randomisation until it has a positive value (to prevent negative values
        # if the sum of the randomized quantitites exceeds the requested
        # base quantity)
        for idx in range(count-1):
            
            # # partial quantitites
            factor = 1 / self.STEP_SIZE
            
            # # get random values for the quantity for every fill
            ticks = (base_qty / self.STEP_SIZE)
            boundary_low = int((ticks / count) * 0.025 * 1000)
            boundary_high = int((ticks / count) * 1.2 * 1000)
            rand_step = int(self.STEP_SIZE * factor) if int(self.STEP_SIZE * factor) > 0 else 1

            try:
                qty = randrange(boundary_low, boundary_high, rand_step) / (factor * 1000) 
            except Exception as e:
                print(e)
                print(f'{factor=} :: {ticks=} :: {boundary_low=} :: {boundary_high=} :: {rand_step=}')
                sys.exit()

            res.append(qty)
            last_qty = round(base_qty - sum(res), self.BASE_ASSET_PRECISION)

        res.append(last_qty)
        return res

    def _get_randomized_prices(self, base_price:float, count:int) -> list:

        prices = []

        for _ in range(count):
            slippage = random() * self.max_slippage
            direction = 'up' if random() > 0.5 else 'down'
            slipppage_amount = round(base_price * slippage, self.TICK_PRECISION)
            if direction == 'up': prices.append(base_price + slipppage_amount)
            else: prices.append(base_price - slipppage_amount)

        return prices

    def _calculate_fill_price(self, fills:list) -> tuple:

        base_qty, quote_qty, price = 0, 0, 0

        for fill in fills:
            base_qty += float(fill['qty'])
            quote_qty += float(fill['price']) * float(fill['qty'])

        base_qty = round(base_qty, self.BASE_ASSET_PRECISION)
        quote_qty = round(quote_qty, self.QUOTE_ASSET_PRECISION)
        price = round(quote_qty / base_qty, self.TICK_PRECISION)        
        
        return price, base_qty, quote_qty

    # -------------------------------------------------------------------------   
    # retrieve the information for the current symbol from the symbol object that
    # should be provided by the calling function. if we only got a symbol name, then
    # this function will ask binance for the informations but this slows the whole 
    # thing down, so it should be avoided
    def _initialize_symbol(self, symbol:Symbol):
        try:
            self.BASE_ASSET = symbol.baseAsset
            self.BASE_ASSET_PRECISION = symbol.baseAssetPrecision
            self.BASE_COMMISSION_PRECISION = symbol.baseCommissionPrecision
            
            self.QUOTE_ASSET = symbol.quoteAsset
            self.QUOTE_ASSET_PRECISION = symbol.quoteAssetPrecision
            self.QUOTE_COMMISSION_PRECISION = symbol.quoteCommissionPrecision
            
            self.COMMISSION_ASSET = self.QUOTE_ASSET
            
            self.TICK_SIZE = symbol.f_priceFilter_tickSize
            self.TICK_PRECISION = symbol.f_tickPrecision
            self.STEP_SIZE = symbol.f_lotSize_stepSize
            self.STEP_PRECISION = symbol.f_stepPrecision
        except:
            self.BASE_ASSET = symbol.base_asset
            self.BASE_ASSET_PRECISION = symbol.base_asset_precision
            self.BASE_COMMISSION_PRECISION = symbol.base_commission_precision
            
            self.QUOTE_ASSET = symbol.quote_asset
            self.QUOTE_ASSET_PRECISION = symbol.quote_asset_precision
            self.QUOTE_COMMISSION_PRECISION = symbol.quote_commission_precision
            
            self.COMMISSION_ASSET = self.QUOTE_ASSET
            
            self.TICK_SIZE = symbol.tick_size
            self.TICK_PRECISION = symbol.tick_precision
            self.STEP_SIZE = symbol.lot_size_step_size
            self.STEP_PRECISION = symbol.lot_size_step_precision
        return 

    # this is only for reference how a 'full' response for different 
    # order types is constructed
    def _dict_builder(self):

        '''
        format of dictionary:

        1) market BUY/SELL orders:

        {'clientOrderId': 'KWzlYZ4xetgBpFTLRqgIor',
        'cummulativeQuoteQty': '10.92400000',
        'executedQty': '10.00000000',
        'fills': [{'commission': '0.01092400',
                    'commissionAsset': 'USDT',
                    'price': '1.09240000',
                    'qty': '10.00000000',
                    'tradeId': 3246868}],
        'orderId': 74960734,
        'orderListId': -1,
        'origQty': '10.00000000',
        'price': '0.00000000',
        'side': 'SELL',
        'status': 'FILLED',
        'symbol': 'BEAMUSDT',
        'timeInForce': 'GTC',
        'transactTime': 1616192949947,
        'type': 'MARKET'}  

        2) STOP LOSS LIMIT ORDER (FULL!)

        {'clientOrderId': 'S8z15KVQHG2U4MqnLfabrW',
        'cummulativeQuoteQty': '0.00000000',
        'executedQty': '0.00000000',
        'fills': [],
        'orderId': 1259087012,
        'orderListId': -1,
        'origQty': '16.40000000',
        'price': '1.15000000',
        'side': 'SELL',
        'status': 'NEW',
        'stopPrice': '1.15100000',
        'symbol': 'ADAUSDT',
        'timeInForce': 'GTC',
        'transactTime': 1618134257805,
        'type': 'STOP_LOSS_LIMIT'}

        '''  

    # -------------------------------------------------------------------------
    def _build_happy_return(self, order_result):

        return {'success' : True,
                'message' : order_result,
                'error' : None,
                'status_code' : 200
                }
    
    def _build_unhappy_return(self, status_code):

        return {'success' : False,
                'message' : None,
                'error' : self.error_message,
                'status_code' : status_code
                }


        