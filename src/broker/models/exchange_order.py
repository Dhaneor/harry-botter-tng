#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon August 01 15:55:23 2022

@author: dhaneor
"""
import sys, os
from dataclasses import dataclass
from typing import Optional
# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------
from util.timeops import unix_to_utc

# ==============================================================================
@dataclass
class ExchangeOrder:

    type: str
    time: int
    symbol: str
    client_order_id: str
    order_id: str
    side: str
    status: str
    executed_quote_qty: float = 0
    executed_base_qty: float = 0
    fill_price: float = 0
    fills: Optional[list] = None
    orig_qty: float = 0
    orig_quote_qty: float = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_human: str = ''
    time_in_force: str = ''
    api_response: Optional[dict] = None

    def __post_init__(self):
        if self.price is None or self.price == 0:
            if self.executed_base_qty != 0 and self.executed_quote_qty != 0:
                self.price = round(
                    self.executed_base_qty / self.executed_quote_qty, 8
                )
            else:
                self.price = self.stop_price

        # if self.type == 'STOP_LOSS_LIMIT':
        #     pprint(self.api_response)
        #     sys.exit()

    def __repr__(self):
        status = self.status

        base_qty = self.executed_base_qty if self.executed_base_qty != 0 \
            else self.orig_qty

        price = self.fill_price if self.fill_price != 0 else self.price

        quote_qty = self.executed_quote_qty if self.executed_quote_qty != 0\
            else self.orig_quote_qty

        # quote_qty = base_qty * self.price if quote_qty == 0 \
        #     else self.orig_quote_qty


        out = f'[{self.time_human}] {status: <10} {self.symbol: <10} ' \
            f'{self.type: >18} {self.side: <5} order for  ' \
            f'{base_qty: <12} at {price: <15} '

        if quote_qty is not None:
            out += f'(funds: {quote_qty:.8f})'

        out += f' (order id: {self.order_id})'

        return out

# ==============================================================================
def build_exchange_order(response:dict) -> ExchangeOrder:

    def _get_fill_price(response:dict) -> float:
        executed_qty =  response.get('executedQty')
        if executed_qty != 0:
            quote_qty = response.get('cummulativeQuoteQty')
            return round(quote_qty / executed_qty, 8)
        else:
            return 0

    for k,v in response.items():
        try:
            response[k] = float(v)
        except Exception as e:
            pass

    if response.get('status') == 'FILLED':
        fill_price = _get_fill_price(response)
    else:
        fill_price = 0.00

    return ExchangeOrder(
        client_order_id=response.get('clientOrderId', ''),
        executed_quote_qty=response.get('cummulativeQuoteQty', ''),
        executed_base_qty=response.get('executedQty', ''),
        fill_price=fill_price,
        fills=response.get('fills'),
        order_id=response.get('orderId'),
        orig_qty=response.get('origQty'),
        orig_quote_qty=response.get('origQuoteOrderQty'),
        price=response.get('price'),
        side=response.get('side'),
        status=response.get('status'),
        stop_price=response.get('stopPrice', None),
        symbol=response.get('symbol'),
        time=response.get('time'),
        time_human=unix_to_utc(response.get('time', 0)),
        time_in_force=response.get('timeInForce'),
        type=response.get('type'),
        api_response=response
    )


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    response = {
        'clientOrderId': 'test',
        'cummulativeQuoteQty': 0.0,
        'executedQty': 0.0,
        'fills': [],
        'icebergQty': 0.0,
        'isWorking': 1.0,
        'orderId': '62e6de703ecb7300016a9e40',
        'orderListId': -1.0,
        'origQty': 14.3772,
        'origQuoteOrderQty': 0.0,
        'price': 0.2,
        'side': 'SELL',
        'status': 'NEW',
        'stopPrice': 0.0,
        'symbol': 'GRT-USDT',
        'time': 1659297392372.0,
        'timeInForce': 'GTC',
        'type': 'LIMIT',
        'updateTime': 1659297392372.0
    }

    print(build_exchange_order(response))


"""
This is how an API response looks like when we request order information:

    {
        'clientOrderId': None,
        'cummulativeQuoteQty': 0.0,
        'executedQty': 0.0,
        'fills': [],
        'icebergQty': 0.0,
        'isWorking': 1.0,
        'orderId': '62e6de703ecb7300016a9e40',
        'orderListId': -1.0,
        'origQty': 14.3772,
        'origQuoteOrderQty': 0.0,
        'price': 0.2,
        'side': 'SELL',
        'status': 'NEW',
        'stopPrice': 0.0,
        'symbol': 'GRT-USDT',
        'time': 1659297392372.0,
        'timeInForce': 'GTC',
        'type': 'LIMIT',
        'updateTime': 1659297392372.0
    }
"""

