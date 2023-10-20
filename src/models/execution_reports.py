#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 13  18 04:55:23 2021

@author: dhaneor
"""

import time

from pprint import pprint

from models.symbol import Symbol
from broker.models.orders import MarketOrder
from mock_responses._message_constructor import MockUSerStreamResponse


# =============================================================================
class ExecutionReport:

    def __init__(self, message=None):

        # determine message type 
        self.original_message = message
        self.ex_type = None

        if message is not None:

            self._type_of_message = None
            self._determine_type_of_message()

            # build from the message type we got
            if self._type_of_message == 'user stream event':
                self._build_from_user_stream_event()
            elif self._type_of_message == 'order response':
                self._build_from_order_response()
            else:
                print(f'{self.type_of_message} message type')
                return

            # set the correct transaction type (see below)
            self._set_type()
            
            del self.original_message

        else:

            return


    def __repr__(self):

        out = f'Execution report for {self.symbol} {self.type} {self.side} order ({self.client_order_id}) - '
        out += f'status {self.current_order_status} '
        out += f'(exc type: {self.ex_type})'

        if self.ex_type == 4:

            out += f': limit price = {self.order_price} '
            out += f'(stop price = {self.stop_price})'
        
        if self.current_order_status == 'FILLED' \
        or self.current_order_status == 'PARTIALLY FILLED':

            quote_qty = float(self.cumulative_quote_asset_transacted_quantity)
            base_qty = float(self.cumulative_filled_quantity)
            price = round(quote_qty / base_qty, 5)

            out += f': {self.cumulative_filled_quantity} for '
            out += f'{self.cumulative_quote_asset_transacted_quantity} '
            out += f'@ {price}'

        if self.order_reject_reason is not None:

            out+= f' Order REJECTED because: {self.order_reject_reason}!'

        return out

    def __eq__(self, other):

        if self.order_id == other.order_id \
        and self.current_order_status == other.current_order_status \
        and self.transaction_time == other.transaction_time:

            return True

        else:

            return False

    # -------------------------------------------------------------------------
    # there are two types of messages: 
    # 1) messages from the user events stream (websocket API)
    # 2) order responses from the REST API
    #
    # this method determines which one we got by checking for keys that are not 
    # present in the other type
    def _determine_type_of_message(self):

        if self.original_message.get('E') is not None: 
            self._type_of_message = 'user stream event'
        elif self.original_message.get('orderId') is not None: 
            self._type_of_message = 'order response'
        else:
            self._type_of_message = 'unknown' 

    def _build_from_user_stream_event(self):

        message = self.original_message

        self.symbol = message.get('s', None) 

        self.order_id = message.get('i', None) 
        self.client_order_id = message.get('c', None) 
        self.trade_id = message.get('t', None)

        self.order_creation_time = message.get('O', None) 
        self.current_order_status = message.get('X', None) 
        self.current_execution_type = message.get('x', None)
        self.time_in_force = message.get('f', None)

        self.side = message.get('S', None)
        self.type = message.get('o', None)

        self.order_quantity = float(message.get('q', None))
        self.quote_order_quantity = float(message.get('Q', None)) 
        self.order_price = float(message.get('p', None))
        self.stop_price = float(message.get('P', None)) 

        self.last_executed_price = float(message.get('L', None)) 
        self.last_executed_quantity = float(message.get('l', None)) 
        self.last_quote_asset_transacted_quantity = float(message.get('Y', None)) 
        self.cumulative_filled_quantity = float(message.get('z', None))
        self.cumulative_quote_asset_transacted_quantity = float(message.get('Z', None))

        self.commission_asset = message.get('N', None) 
        self.commission_amount = round(float(message.get('n', None)), 8) 

        self.event_time = message.get('E', None) 
        self.transaction_time = message.get('T', None) 

        self.is_this_trade_maker_side = True if message.get('m', False) == 'true' else False
        self.is_the_message_on_the_book = True if message.get('w', False) == 'true' else False
        self.iceberg_quantity = float(message.get('F', 0.00))

        self.order_reject_reason = None if message.get('r') == 'NONE' else message.get('r')


    def _build_from_order_response(self):

        message = self.original_message

        fills = message.get('fills')
        if len(fills) > 0:
            fill_results = self._get_fill_results(message.get('fills'))
        else:
            fill_results = {'commission' : 0.00,
                            'commission_asset' : None,
                            'price' : 0.00,
                            'qty' : 0.00,
                            'quote qty' : 0.00,
                            'trade id' : None
                  }

        self.symbol = message.get('symbol', None) 

        self.order_id = message.get('orderId', None) 
        self.client_order_id = message.get('clientOrderId', None) 
        self.trade_id = fill_results.get('trade id')

        self.order_creation_time = message.get('transactTime', 0)
        self.current_order_status = message.get('status', '')
        self.current_execution_type = message.get('status', '')
        self.time_in_force = message.get('timeInForce', '')

        self.side = message.get('side', None)
        self.type = message.get('type', None)

        self.order_quantity = float(message.get('origQty', 0.00))
        self.quote_order_quantity = fill_results.get('quote_qty') 
        self.order_price = float(message.get('price', 0.00))
        self.stop_price = float(message.get('stopPrice', 0.00)) 

        self.last_executed_price = fill_results.get('price')
        self.last_executed_quantity = fill_results.get('qty')
        self.last_quote_asset_transacted_quantity = fill_results.get('quote qty')
        self.cumulative_filled_quantity = float(message.get('executedQty', 0.00))
        self.cumulative_quote_asset_transacted_quantity = float(message.get('cummulativeQuoteQty', 0.00))

        self.commission_asset = fill_results.get('commission asset') 
        self.commission_amount = fill_results.get('commission')

        self.event_time = message.get('transactTime', None) 
        self.transaction_time = message.get('transactTime', None) 

        if self.type == ('LIMIT' or 'STOP_LOSS_LIMIT' or 'TAKE_PROFIT_LIMIT'):
            self.is_this_trade_maker_side = True
            if self.current_order_status == ('NEW' or 'PARTIALLY FILLED'):
                self.is_the_message_on_the_book = True
        else: 
            self.is_this_trade_maker_side = False
            self.is_the_message_on_the_book = False

        self.iceberg_quantity = 0.00

        self.order_reject_reason = None

    def _get_fill_results(self, fills : list):

        commission = 0
        commission_asset = ''
        price = 0.00
        qty = 0.00
        quote_qty = 0.00
        trade_id = ''

        for fill in fills:

            commission += float(fill['commission'])
            commission_asset = fill['commissionAsset']
            quote_qty = float(fill['qty']) * float(fill['price'])
            qty += float(fill['qty'])
            trade_id = fill['tradeId']

        price = round(quote_qty / qty, 8)

        result = {'commission' : commission,
                  'commission asset' : commission_asset,
                  'price' : price,
                  'qty' : qty,
                  'quote qty' : quote_qty,
                  'trade id' : trade_id
                  }

        return result



    # -------------------------------------------------------------------------
    # set a numerical value for the type of transaction. this is just used to 
    # hide that logic from the main class and can be accessed/used there
    #
    # 1: BUY order FILLED [MARKET or LIMIT]
    # 2: SELL ORDER FILLED [MARKET or LIMIT]
    # 3: STOP LOSS LIMIT order CANCELED
    # 4: STOP LOSS LIMIT order CREATED (NEW)
    # 5: STOP LOSS triggered
    #
    def _set_type(self):

        if self.side == 'BUY' \
        and self.current_order_status == 'FILLED' \
        and self.type != 'STOP_LOSS_LIMIT':
            self.ex_type = 1
        
        elif self.side == 'SELL' \
        and self.current_order_status == 'FILLED' \
        and self.type != 'STOP_LOSS_LIMIT':
            self.ex_type = 2

        elif self.type == 'STOP_LOSS_LIMIT' and self.current_order_status == 'CANCELED':
            self.ex_type = 3

        elif self.type == 'STOP_LOSS_LIMIT' and self.current_order_status == 'NEW':
            self.ex_type = 4

        elif self.type == 'STOP_LOSS_LIMIT' and self.current_order_status == 'FILLED':
            self.ex_type = 5

    # -------------------------------------------------------------------------
    # this is an original response, just kept here for reference
    def response(self):

        '''
        This is the format of a USER STREAM MESSAGE for an order action

        {'C': '',
            'E': 1618268329645,
            'F': '0.00000000',
            'I': 2651735495,
            'L': '0.00000000',
            'M': False,
            'N': None,
            'O': 1618268329644,
            'P': '0.00000000',
            'Q': '0.00000000',
            'S': 'SELL', // side
            'T': 1618268329644,
            'X': 'NEW',
            'Y': '0.00000000',
            'Z': '0.00000000',
            'c': 'web_0b1087f19b254ba288fb7e105a651c74', // client order id
            'e': 'executionReport',
            'f': 'GTC',
            'g': -1,
            'i': 1269086272,
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

            here is another one with comments on what the keys/fields mean:

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

        '''

        '''
        Here are two examples of the format for a response from the REST API for 
        an order that was sent by the client:

        {'clientOrderId': 'SIpAS2YcuR1xWlkg6ROyN1',
            'cummulativeQuoteQty': '13.32000000',
            'executedQty': '6.00000000',
            'fills': [{'commission': '0.00600000',
                        'commissionAsset': 'ADA',
                        'price': '2.22000000',
                        'qty': '6.00000000',
                        'tradeId': 294075127}],
            'orderId': 2422371930,
            'orderListId': -1,
            'origQty': '6.00000000',
            'price': '0.00000000',
            'side': 'BUY',
            'status': 'FILLED',
            'symbol': 'ADAUSDT',
            'timeInForce': 'GTC',
            'transactTime': 1633107293430,
            'type': 'MARKET'}

        {'clientOrderId': 'olIbewRp17FWUZLP8A0nDE',
            'cummulativeQuoteQty': '0.00000000',
            'executedQty': '0.00000000',
            'fills': [],
            'orderId': 2422479475,
            'orderListId': -1,
            'origQty': '6.00000000',
            'price': '2.21000000',
            'side': 'SELL',
            'status': 'NEW',
            'stopPrice': '2.21200000',
            'symbol': 'ADAUSDT',
            'timeInForce': 'GTC',
            'transactTime': 1633109755687,
            'type': 'STOP_LOSS_LIMIT'}

        '''
        pass 



# =============================================================================

if __name__ == '__main__':

    st = time.time()



    symbol_name = 'ADAUSDT'
    symbol = Symbol(symbol_name)

    # order = MarketOrder(symbol=symbol,
    #                     market='SPOT',
    #                     side='SELL',
    #                     base_qty='50.00',
    #                     )

    # erb = MockUSerStreamResponse(symbol, symbol.quoteAsset)
    # msg = erb.executed_order_message(order)

    msg = {'clientOrderId': 'S8z15KVQHG2U4MqnLfabrW',
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

    msg = {'clientOrderId': 'KWzlYZ4xetgBpFTLRqgIor',
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

    er_obj = ExecutionReport(msg)

    pprint(er_obj.__dict__)

    pprint(er_obj)
    print('-'*80)
    print(f'{round(time.time() - st, 2)}s')