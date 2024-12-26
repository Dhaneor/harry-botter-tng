#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:05:18 2021

@author: dhaneor
"""
import string
import time
import sys
from random import choice
from pprint import pprint
from typing import Union, List

from broker.models.symbol import Symbol
from exchange.util.formatting import scientific_to_str as sts


# =============================================================================
class ExecutionReport:

    def __init__(self, message: dict):

        self.symbol: str

        # determine message type
        self.original_message: dict = message
        self._type_of_message: str
        self.successful_order: bool
        self.current_order_status: str
        self.execution_errors: list = []

        # let's see if we got a user stream event or an order response event
        self._determine_type_of_message()

        # build from the message/event type we got
        if self._type_of_message == 'user stream event':
            self._build_from_user_stream_event()
        elif self._type_of_message == 'order response':
            self._build_from_order_response()
        else:
            print(f'{self._type_of_message} message type')
            return

        self.ex_type = None
        self._set_type()

    def __repr__(self):

        type =  '.' if self._type_of_message == 'order response' else '_'

        out = f'{type}Execution Report for {self.symbol} {self.type} '
        out += f'{self.side} order ({self.client_order_id}) - '
        out += f'status {self.current_order_status} (exc type: {self.ex_type})'

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

        if len(self.execution_errors) > 0:

            for e in self.execution_errors: print(f"\n{e['message']}\n")

        return out

    def __eq__(self, other):

        if self.client_order_id == other.client_order_id \
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
        # event type: user stream event
        if 'E' in self.original_message:
            self._type_of_message = 'user stream event'
        # event type: structured order response (from exchange client)
        elif 'message' in self.original_message:
            self._type_of_message = 'order response'
        else:
            print('failed to determine message type for:')
            pprint(self.original_message)
            sys.exit()
            self._type_of_message = 'unknown'

    def _build_from_user_stream_event(self):

        message = self.original_message
        execution_error = message.get('r')

        if execution_error == 'NONE':

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

            self.successful_order = True


    def _build_from_order_response(self):
        try:
            message = self.original_message['message']
            success = self.original_message['success']
        except:
            pass
            return

        if success:

            fills = message.get('fills', [])

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
            self.successful_order = True

        else:
            self.execution_errors.append(self.original_message)

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

        if not self.original_message.get('success', False):
            self.ex_type = -1
            return

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
#                                   ORDERS                                    #
# =============================================================================
class Order:

    markets = ['SPOT', 'CROSS MARGIN', 'ISOLATED MARGIN', 'FUTURES']
    sides = ['BUY', 'SELL']

    def __init__(self, **kwargs):
        """Base Class for order objects

        :param symbol: a symbol object
        :type symbol: Symbol
        :param exchange: name of the exchange ('binance'/'kucoin')
        :type exchange: str
        :param market: 'SPOT', 'CROSS MARGIN' or 'ISOLATED MARGIN'
        :type market: str
        :param side: 'BUY' or 'SELL'
        :type side: str
        :param base_qty: amount of base asset (optional if quote asset is given)
        :type base_qty: str
        :param quote_qty: amount of quote asset (optional if base asset is given)
        :type quote_qty: str
        :param last_price: the last/current price for the symbol
        :type last_price: float
        """
        self.order_id: str
        self.client_order_id = self._build_client_order_id()

        self.symbol: Symbol = kwargs['symbol'] # symbol object
        self.exchange: str = kwargs.get('exchange', '')
        self.market: str = kwargs.get('market', '')
        self.side: str = kwargs.get('side', '')
        self.base_qty: Union[float, None] = kwargs.get('base_qty')
        self.quote_qty: Union[float, None] = kwargs.get('quote_qty')
        self.auto_borrow: bool = kwargs.get('auto_borrow', False)

        self.type: str
        self.base_qty_net: float

        self.stop_price: float
        self.limit_price: float

        self.creation_time: int = int(time.time() * 1000)
        self.execution_times: dict = {'initiated' : 0,
                                'approved' : 0,
                                'created' : 0,
                                'executed' : 0,
                                'rejected' : 0,
                                'failed' : 0
                                }

        self._status: str # INITIATED. APPROVED, PENDING, CREATED,
                           # EXECUTED, REJECTED, FAILED
        self.valid: bool = False # False until order is validated
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

        self.execution_errors = []
        self.execution_warnings = []
        self.all_execution_reports = []
        self.last_execution_report: ExecutionReport
        self.result: Union[dict, None] = None

        self.last_price: Union[float, None] = kwargs.get('last_price')

        self.fill_price: Union[float, None] = None
        self.slippage: float
        self.slippage_per_cent: float

        self.commission_base: float
        self.commission_quote: float
        self.commission_bnb: float
        self.commission_per_cent: float
        self.sum_commissions_as_quote: float


    def __repr__(self):

        if self.status:
            exc_time = self.execution_times[self.status.lower()]
        else:
            exc_time = 0

        res = f'{self.type} {self.side} order on {self.symbol.name}'\
            f' {self.market} market {self.status} for '

        if self.base_qty is not None:
            res += f'{sts(self.base_qty)} {self.symbol.base_asset} '
        elif self.quote_qty:
            res += f' {round(self.quote_qty, self.symbol.tick_precision)} '\
                f'{self.symbol.quote_asset} '

            if self.last_price:
                base_qty = float(self.quote_qty) / self.last_price
                base_qty = round(base_qty, self.symbol.lot_size_step_precision)
                res += f' (~{sts(base_qty)} {self.symbol.base_asset}) '

        if self.type == 'MARKET' and self.last_price:
            res += f'[last price: {sts(self.last_price)} {self.symbol.quote_asset}]'

        if 'STOP' in self.type:
            res += f' - stop price: {self.stop_price} '

        if self.type == 'LIMIT':
            res += f':: limit price: {self.limit_price}'

        res += f'\t({self.client_order_id} - exc time: {exc_time}ms)'

        if self.auto_borrow:
            res += ' [AUTO_BORROW]\n'
        else:
            res += '\n'

        if self.type == 'STOP_LOSS_LIMIT':
            res += f'stop price: {self.stop_price}\tlimit price: {self.limit_price}\n'

            if self.result:
                order_id = self.result.get('orderId')
                res += f'order ID: {order_id}\n'

        if self.validation_warnings:
            for w in self.validation_warnings: res += f'{w} \n'

        if self.status == 'REJECTED':
            for e in self.validation_errors: res += f'{e} \n'

        if self.status == 'FILLED' and self.fill_price:
            res += f'expected price: {self.last_price} {self.symbol.quote_asset} \t\t\t'
            res += f'realized price: {self.fill_price} {self.symbol.quote_asset} \n'
            res += f'slippage:       {sts(self.slippage)} {self.symbol.quote_asset} ({self.slippage_per_cent}%) \t\t'
            res += f'commission:     {self.sum_commissions_as_quote} {self.symbol.quote_asset} ({self.commission_base})\n'

        if self.status == 'FAILED':
            for e in self.execution_errors:
                res += f'{e} \n'

        return res

    @property
    def execution_report(self):
        return self.last_execution_report

    @execution_report.setter
    def execution_report(self, event: dict):

        self.last_execution_report = ExecutionReport(event)

        if len(self.last_execution_report.execution_errors) > 0:
            self.execution_errors = self.last_execution_report.execution_errors
            self.status = 'FAILED'

        self.status = self.last_execution_report.current_order_status
        self.all_execution_reports.append(self.last_execution_report)

        self.result = self.last_execution_report.original_message['message']

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status : str):

        valid_status = ['INITIATED', 'APPROVED', 'PENDING', 'NEW', 'CREATED',
                        'ABORTED','EXECUTED', 'FILLED', 'REJECTED', 'FAILED',
                        'CANCELED, CANCELLED']

        if status in valid_status:

            self._status = status
            # print('setting status to ', self._status)

            et = round((time.time()*1000 - self.creation_time), 2)
            self.execution_times[status.lower()] = et

        if self.status == 'APPROVED': self.execution_times['rejected'] = 0

        if self.status == 'FILLED':
            self._calculate_result_stats()

        return


    # -------------------------------------------------------------------------
    def _order_is_valid(self):

        self.status = 'REJECTED'

        if isinstance(self.market, str):
            self.market = self.market.upper()
        if isinstance(self.side, str):
            self.side = self.side.upper()

        # check if MARKET is a valid value
        if self.market is None or self.market not in Order.markets:
            self.validation_warnings.append((f'Invalid market: {self.market}'))

        # check if SIDE is a valid value
        if self.side is None or self.side not in Order.sides:
            self.validation_warnings.append((f'Invalid side: {self.side}'))

        if self.last_price is None:
            self.validation_errors.append((f'Missing parameter: last_price'))

        # change order status if there were no validation errors
        if not self.validation_errors:
            self.status = 'INITIATED'
            self.valid = True
            return True

        return False

    # -------------------------------------------------------------------------
    def _build_client_order_id(self):
        let_num = string.ascii_letters + string.digits
        return ''.join([choice(let_num) for _ in range(22)])

    def _calculate_result_stats(self):

        if self.result is None:
            return

        base_asset = self.symbol.base_asset
        quote_asset = self.symbol.quote_asset

        base_precision = self.symbol.base_asset_precision
        quote_precision = self.symbol.quote_asset_precision
        tick_precision = self.symbol.tick_precision

        # ---------------------------------------------------------------------
        # calculate the fill price
        base_qty = float(self.result.get('executedQty', 0))
        cum_quote_qty = float(self.result.get('cummulativeQuoteQty', 0))

        self.fill_price = round(cum_quote_qty / base_qty, tick_precision)

        # calculate the slippage - absolute and in per cent
        if not self.last_price:
            return

        if self.side == 'BUY' and self.type == 'MARKET':
            self.slippage = self.last_price - self.fill_price
            self.slippage_per_cent = (self.fill_price / self.last_price - 1) * 100

        elif self.side == 'SELL' and self.type == 'MARKET':
            self.slippage = self.last_price - self.fill_price
            self.slippage_per_cent = (self.fill_price / self.last_price - 1) * 100

        elif self.side == 'BUY' and self.type == 'STOP_LOSS_LIMIT':
            self.slippage = self.stop_price - self.fill_price
            self.slippage_per_cent = (self.fill_price / self.stop_price - 1) * 100

        elif self.side == 'SELL' and self.type == 'STOP_LOSS_LIMIT':
            self.slippage = self.stop_price - self.fill_price
            self.slippage_per_cent = (self.fill_price / self.stop_price - 1) * 100

        self.slippage = round(self.slippage, tick_precision)
        self.slippage_per_cent = round(self.slippage_per_cent, 4)

        # ---------------------------------------------------------------------
        # cycle through the fills and determine net quantity and commissions
        fills = self.result.get('fills')

        if not fills:
            return

        real_price, net_qty = 0, 0
        commission_base, commission_quote, commission_bnb, cum_quote_comm = 0, 0 ,0, 0

        for fill in fills:

            # -------------------------------
            # example for one fill:
            #
            # {'commission': 0.0592,
            #     'commissionAsset': 'ADA',
            #     'price': '1.22232031',
            #     'qty': '59.2',
            #     'tradeId': '9028182'}

            price = float(fill.get('price', 0))
            qty = float(fill.get('qty', 0))
            commission = float(fill.get('commission', -1))

            # calculation for BUY orders
            if self.side == 'BUY':
                if fill.get('commissionAsset') == base_asset:
                    net_qty += qty - commission
                    commission_base += commission
                    cum_quote_comm += commission * price
                else:
                    ca = fill.get('commissionAsset')
                    print(f'{ca=} <> {base_asset}')

            # calculation for SELL orders
            elif self.side == 'SELL':
                if fill.get('commissionAsset') == quote_asset:
                    net_qty += qty - commission
                    commission_quote += commission
                    cum_quote_comm += commission
                else:
                    ca = fill.get('commissionAsset')
                    print(f'{ca=} <> {quote_asset}')


        # calculate 'real' fill price for the order
        if net_qty != 0:
            self.real_price = round(cum_quote_qty / net_qty, quote_precision)

        # calculate the commission expressed in quote asset amount for buy orders
        # (where the commission is deducted from the base asset received)
        commission_base = round(commission_base, base_precision)
        commission_quote = round(commission_quote, quote_precision)
        cum_quote_comm = round(cum_quote_comm, quote_precision)

        self.base_qty_net = round(net_qty, base_precision)
        self.commission_base = commission_base
        self.commission_quote = commission_quote
        # self.commission_bnb = commission_bnb
        self.sum_commissions_as_quote = cum_quote_comm

        # calculate the commission in per cent of the transacted quote asset quantity
        self.commission_per_cent = round((cum_quote_comm / cum_quote_qty) * 100, 2)

        return

# =============================================================================
#                               MARKET ORDER requests                         #
# =============================================================================
class MarketOrder(Order):

    def __init__(self, **kwargs):

        Order.__init__(self, **kwargs)
        self.type = 'MARKET'

        self.valid = self._order_is_valid()
        self.valid = self._is_valid_market_order()

    def _is_valid_market_order(self):

        self.status = 'REJECTED'

        # check if there is a value for either BASE QUANTITY or QUOTE QUANTITY
        if self.base_qty is None and self.quote_qty is None:
            self.validation_errors.append(
                f'misssing parameter: base quantity / quote quantity (both are None)'
                )
            return False

        # convert 'base quantity' to float if necessary
        if self.base_qty is not None and not isinstance(self.base_qty, float):
            try:
                self.base_qty = float(self.base_qty)
            except:
                self.validation_errors.append(
                    (f'Invalid base quantity type: {type(self.base_qty)}'))

        # convert 'quote quantity' to float if necessary
        if self.quote_qty is not None and not isinstance(self.quote_qty, float):
            try:
                self.quote_qty = float(self.quote_qty)
            except:
                self.validation_errors.append(
                    (f'Invalid quote quantity type: {type(self.quote_qty)}'))

        # change order status if there were no validation errors
        if not self.validation_errors:
            # print('first test successful...')
            self.status = 'INITIATED'
            self.valid = True
            return True

        return False

class StopMarketOrder(MarketOrder):

    def __init__(self, **kwargs):

        MarketOrder.__init__(self, **kwargs)
        self.type = 'STOP_LOSS'
        self.stop_price: Union[float, None] = kwargs.get('stop_price')

        self._is_valid_stop_market_order()

    # -------------------------------------------------------------------------
    # some additional checks specific to STOP LIMIT orders
    def _is_valid_stop_market_order(self):

        already_rejected = True if self.valid == False else False

        self.status = 'REJECTED'
        self.valid = False

        # check if we really have a stop price
        if self.stop_price is None:
            reason = f'Missing stop price ({self.stop_price})'
            self.validation_errors.append(reason)

        # check if STOP PRICE type is valid or at least
        # can be converted to a float number
        if self.stop_price is not None and not isinstance(self.stop_price, float):
            try:
                self.stop_price = float(self.stop_price)
            except:
                reason = f'Invalid stop price {self.stop_price}'
                self.validation_errors.append(reason)

        # approve order and set status to INITIATED if there were no errors
        if not self.validation_errors and not already_rejected:
            self.status = 'INITIATED'
            self.valid = True
            return True

        return False


# =============================================================================
#                               LIMIT ORDER requests                          #
# =============================================================================
class LimitOrder(Order):

    def __init__(self, **kwargs):

        Order.__init__(self, **kwargs)

        self.type = 'LIMIT'
        self.limit_price: float = kwargs.get('limit_price', 0)

        self._is_valid_limit_order()

    def _is_valid_limit_order(self):

        self.status = 'REJECTED'
        self.valid = False

        # check if we have a limit price with the correct type or a type, that
        # can be converted to float
        if self.limit_price is not None and not isinstance(self.limit_price, float):
            try:
                self.limit_price = float(self.limit_price)
            except Exception:
                self.validation_errors.append(
                    f'invalid limit price type: {type(self.limit_price)}'
                )

        if self.limit_price is None:
            self.validation_errors.append(
                f'missing parameter: limit price'
            )

        # convert 'base quantity' to float if necessary
        if self.base_qty is not None and not isinstance(self.base_qty, float):
            try:
                self.base_qty = float(self.base_qty)
            except:
                self.validation_errors.append(
                    f'Invalid base quantity type: {type(self.base_qty)}'
                )

        if not self.validation_errors:
            self.status = 'INITIATED'
            self.valid = True

class StopLimitOrder(LimitOrder):

    def __init__(self, **kwargs):

        LimitOrder.__init__(self, **kwargs)

        self.type = 'STOP_LOSS_LIMIT'
        self.limit_price = float(kwargs.get('limit_price', None))
        self.stop_price = float(kwargs.get('stop_price', None))
        # self.last_price = float(kwargs.get('last_price', 0))

        self._order_is_valid()
        self._is_valid_stop_limit_order()

    # -------------------------------------------------------------------------
    # some additional checks specific to STOP LIMIT orders
    def _is_valid_stop_limit_order(self):

        if self.valid == False: already_rejected = True
        else: already_rejected = False

        self.status = 'REJECTED'
        self.valid = False

        # check if we really have a stop price and a limit price
        if self.limit_price is None:
            reason = f'Missing limit price ({self.limit_price})'
            self.validation_errors.append(reason)

        if self.stop_price is None:
            reason = f'Missing stop price ({self.stop_price})'
            self.validation_errors.append(reason)

        # check if stop price and and limit price type is valid or at least
        # can be converted to a float number
        if self.limit_price is not None and not isinstance(self.limit_price, float):
            try:
                self.limit_price = float(self.limit_price)
            except:
                reason = f'Invalid limit price ({self.limit_price})'
                self.validation_errors.append(reason)

        if self.stop_price is not None and not isinstance(self.stop_price, float):
            try:
                self.stop_price = float(self.stop_price)
            except:
                reason = f'Invalid stop price ({self.stop_price})'
                self.validation_errors.append(reason)

        # approve order and set status to INITIATED if there were no errors
        if not self.validation_errors and not already_rejected:
            self.status = 'INITIATED'
            self.valid = True
            return True

        return False

# =============================================================================
class CancelOrder(Order):

    def __init__(self, **kwargs):

        Order.__init__(self, **kwargs)

        self.order_id = kwargs.get('order_id', None)
        self.orig_client_order_id = kwargs.get('orig_client_order_id')

        # print('CANCEL ORDER REQUEST:')
        # pprint(self.__dict__)
