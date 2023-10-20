#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:01:14 2022

@author: dhaneor
"""
from decimal import Decimal
from typing import Union

try:
    from .timeops import unix_to_utc
except:
    pass

def _get_market_order():
    return {'clientOrderId': '66fa60d0-9369-11ec-a2cd-1e00623eee81',
            'cummulativeQuoteQty': '9.9932358968',
            'executedQty': '54.0982',
            'fills': [],
            'icebergQty': '0.00000000',
            'isWorking': False,
            'orderId': '6214187135264300014841a8',
            'orderListId': -1,
            'origQty': '54.0982',
            'origQuoteOrderQty': '0',
            'price': '0',
            'side': 'SELL',
            'status': 'FILLED',
            'stopPrice': '0',
            'symbol': 'XLM-USDT',
            'time': 1645484145447,
            'timeInForce': 'GTC',
            'type': 'MARKET',
            'updateTime': 1645484145447}
    
def _get_stop_limit_order():
    return {'clientOrderId': 'RV9SRdlQvatep1nsNX64vH',
            'cummulativeQuoteQty': '0',
            'executedQty': '0',
            'fills': [],
            'icebergQty': '0.00000000',
            'isWorking': False,
            'orderId': 'vs8nqogk31lv89e6000m376d',
            'orderListId': -1,
            'origQty': '54.0982',
            'origQuoteOrderQty': None,
            'price': '0.157122',
            'side': 'SELL',
            'status': 'CANCELED',
            'stopPrice': '0.166364',
            'symbol': 'XLM-USDT',
            'time': 1645484139870,
            'timeInForce': 'GTC',
            'type': 'STOP_LOSS_LIMIT',
            'updateTime': 1645484139870}


# =============================================================================
def human_order_response(response:dict) -> str:
    
    if not response:
        return
    
    _time = unix_to_utc(response.get('updateTime'), 0)
    symbol = response.get('symbol')
    type = response.get('type')
    side = response.get('side')
    base_qty = float(response.get('executedQty'), 0)
    quote_qty = float(response.get('cummulativeQuoteQty'), 0)
    status = response.get('status')
    _precision = len(response.get('icebergQty', '0.0').split('.')[1])
        
    try:
        price = round(quote_qty / base_qty, _precision)
    except:
        price = 0.00

    # .........................................................................
    human = f"[{_time}] {symbol} {type} {side} order for "
    
    if type == 'MARKET':
        human += f"{base_qty} - price: {price}"
    
    if 'LIMIT' in type:
        orig_base_qty = float(response.get('origQty'))
        percent_filled = round(base_qty / orig_base_qty, _precision)
        base_qty = orig_base_qty
        price = response.get('price')
        
        human += f"{base_qty} "
        
        if 'FILLED' in status:
            human += f"({percent_filled}% filled) "
        
        human += f"- price: {price}"
    
    if 'STOP' in type:
        stop_price = response.get('stopPrice')
        human += f" :: stop price: {stop_price}"
        
    human += f' [status: {status}]'

    return human 

def scientific_to_str(value: Union[float, int, str]) -> str:
    if value is not None:
        res = "{:.8f}".format(float(value))  
        while res[-1] == '0':
            res = res[:-1]  
    else:
        res = value
    return res



# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    print(scientific_to_str(value=0.00002388))
    print(scientific_to_str(value=1.878e-05))
    
    # print(human_order_response(_get_market_order()))
    # print(human_order_response(_get_stop_limit_order()))
    


