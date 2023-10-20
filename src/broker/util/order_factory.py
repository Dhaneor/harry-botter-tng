#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nonv 23 19:30:20 2022

@author dhaneor
"""
from typing import Union

from staff.saturn import Saturn
from broker.models.orders import (
    Order, MarketOrder, StopMarketOrder, LimitOrder, StopLimitOrder
)
from ..models.symbol import Symbol

# =============================================================================
class OrderFactory:
    """Factory class that builds valid orders of different types.
    """
    def __init__(self):
        self.symbol: Symbol
        self.exchange: str
        self.market: str
        self._default_order_type: str = 'market' # 'market' or 'limit
        
        self.saturn = Saturn()

    # --------------------------------------------------------------------------
    def build_buy_order(
        self, 
        symbol: Symbol,
        type: Union[str, None]=None,
        base_qty: Union[float, None]=None, 
        quote_qty: Union[float, None]=None,
        limit_price: Union[float, None]=None, 
        last_price: Union[float, None]=None
        ) -> Order:
        """Builds and returns a BUY order (market or limit order).

        :param symbol: name of a symbol or a Symbol object
        :type symbol: Union[str, Symbol]
        :param type: 'market' or 'limit' order type, defaults to None
        :type type: str, optional
        :param base_qty: base asset quantity , defaults to None
        :type base_qty: float, optional
        :param quote_qty: quote asset quantity, defaults to None
        :type quote_qty: float, optional
        :param limit_price: limit price (only for limit orders), 
                            defaults to None
        :type limit_price: float, optional
        :param last_price: last traded price for symbol, defaults to None
        :type last_price: float, optional
        :return: a validated (or rejected) order object
        :rtype: Order
        """
        self._set_symbol(symbol)
        type = self._default_order_type if not type else type
        
        if type == 'market':
            order = MarketOrder(symbol=symbol,
                                side='BUY',
                                exchange=self.exchange,
                                market=self.market,
                                base_qty=base_qty,
                                quote_qty=quote_qty,
                                last_price=last_price
                                ) 
                    
        elif type == 'limit':
            order = LimitOrder(symbol=self.symbol,
                               exchange=self.exchange,
                               market=self.market,
                               side='BUY',
                               base_qty=base_qty,
                               quote_qty=quote_qty,
                               limit_price = limit_price,
                               last_price=last_price)
        else:
            raise ValueError('invalid order type: {type}')
            
        if order.status == 'REJECTED': 
            return order
        else:
            return self.saturn.cm.validate(order) 
 
    def build_sell_order(
        self, 
        symbol: Symbol,
        type: Union[str, None]=None,
        base_qty: Union[float, None]=None, 
        quote_qty: Union[float, None]=None,
        limit_price: Union[float, None]=None, 
        last_price: Union[float, None]=None
        ) -> Order:
        """Builds and returns a BUY order (market or limit order).

        :param symbol: name if a symbol or a Symbol object
        :type symbol: Union[str, Symbol]
        :param type: 'market' or 'limit' order type, defaults to None
        :type type: str, optional
        :param base_qty: base asset quantity , defaults to None
        :type base_qty: float, optional
        :param quote_qty: quote asset quantity, defaults to None
        :type quote_qty: float, optional
        :param limit_price: limit price (only for limit orders), 
                            defaults to None
        :type limit_price: float, optional
        :param last_price: last traded price for symbol, defaults to None
        :type last_price: float, optional
        :return: a validated (or rejected) order object
        :rtype: Order
        """
        self._set_symbol(symbol)
        type = self._default_order_type if not type else type
        
        if type == 'market':
            order = MarketOrder(symbol=self.symbol,
                                side='SELL',
                                exchange=self.exchange,
                                market=self.market,
                                base_qty=base_qty,
                                quote_qty=quote_qty,
                                last_price=last_price
                                ) 
        
        else:  
            order = LimitOrder(symbol=self.symbol,
                               exchange=self.exchange,
                               market=self.market,
                               side='SELL',
                               base_qty=base_qty,
                               quote_qty=quote_qty,
                               limit_price = limit_price,
                               last_price=last_price)
            
        if order.status == 'REJECTED': 
            return order
        else:
            return self.saturn.cm.validate(order) 
    
    def build_long_stop_order(
        self, 
        symbol:Symbol,
        base_qty: float, 
        stop_price: float, 
        last_price:float, 
        limit_price: Union[float, None]=None,
        type: Union[str, None]=None,  
        ) -> Order:
        
        self._set_symbol(symbol)
        
        if not type:
            type = 'limit' if limit_price else 'market'
        
        if type == 'limit':
            order = StopLimitOrder(symbol=self.symbol,
                                   exchange=self.exchange,
                                   market=self.market,
                                   side='SELL',
                                   base_qty=base_qty,
                                   stop_price=stop_price,
                                   limit_price=limit_price,
                                   last_price=last_price)
        else:
            order = StopMarketOrder(symbol=self.symbol,
                                    exchange=self.exchange,
                                    market=self.market,
                                    side='SELL',
                                    base_qty=base_qty,
                                    stop_price=stop_price,
                                    last_price=last_price)
            
        if order.status == 'REJECTED': 
            return order
        else:
            return self.saturn.cm.validate(order)
               
    def build_short_stop_order(
        self, 
        symbol: Symbol, 
        base_qty: float, 
        stop_price: float, 
        last_price:float, 
        limit_price: Union[float, None]=None, 
        type: Union[str, None]=None 
        ) -> Order:
        
        self._set_symbol(symbol)
        
        if not type:
            type = 'limit' if limit_price else 'market'
        
        if type == 'limit':
            order = StopLimitOrder(symbol=symbol,
                                   exchange=self.exchange,
                                   market=self.market,
                                   side='BUY',
                                   base_qty=base_qty,
                                   stop_price=stop_price,
                                   limit_price=limit_price,
                                   last_price=last_price)
        else:
            order = StopMarketOrder(symbol=self.symbol,
                                    exchange=self.exchange,
                                    market=self.market,
                                    side='BUY',
                                    base_qty=base_qty,
                                    stop_price=stop_price,
                                    last_price=last_price)
            
        if order.status == 'REJECTED': 
            return order
        else:
            return self.saturn.cm.validate(order)
            
    # --------------------------------------------------------------------------
    def _set_symbol(self, symbol: Symbol):

        if isinstance(symbol, Symbol):
            self.symbol = symbol
        else:
            raise TypeError(
                f"'symbol' must be <str> or <Symbol> but was {type(symbol)}")
            
        self.exchange = 'kucoin' if '-' in self.symbol.name else 'binance'
        self.market = 'CROSS MARGIN' if self.exchange == 'kucoin' else 'SPOT'
        
