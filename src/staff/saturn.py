#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""
import math
from typing import Union

from broker.models.orders import (
    MarketOrder, StopMarketOrder, LimitOrder, StopLimitOrder
)

Order = Union[MarketOrder, StopMarketOrder, LimitOrder, StopLimitOrder]
StopOrder = Union[StopMarketOrder, StopLimitOrder] 

# =============================================================================
class ComplianceManager:

    def __init__(self, **kwargs):
        self.name = 'Compliance Manager'
        self.last_price = 0 

    # -------------------------------------------------------------------------
    def validate(self, order : Order) -> Order:
        return self.check_order(order)

    def check_order(self, order: Order) -> Order:

        self.symbol = order.symbol
        
        order = self._check_parameter_type(order)

        if order.status == 'REJECTED':
            return order
        
        if order.last_price != 0 and order.last_price is not None: 
            self.last_price = order.last_price
        else:
            order.validation_errors.append(
                '[ComplianceManager] need to get last price from the exchange!'\
                    ' include last price for faster processing!'
            )
            order.status = 'REJECTED'
            return order


        order = self._check_order_type(order)
        if order.status == 'REJECTED': 
            return order
        
        market_order_types = ['MARKET', 'STOP_LOSS', 'STOP_MARKET']

        limit_order_types = ['LIMIT' ,'LIMIT_MAKER', 
                             'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']
        
        if order.type in market_order_types:
            
            checks = [self._check_base_or_quote,
                      self._check_position_size_market_order,
                      self._check_step_size,
            ]
            
            for check in checks:
                order = check(order)
                if order.status == 'REJECTED': 
                    return order

        elif order.type in limit_order_types:
            
            checks = [
                self._check_base_or_quote,
                self._check_prices,
                self._check_position_size_limit_order,
                self._check_step_size,
            ]
            
            for check in checks:
                order = check(order)
                if order.status == 'REJECTED': 
                    return order

        if order.type and 'STOP' in order.type:
            order = self._check_stop_prices(order) #type:ignore
            if order.status == 'REJECTED':
                return order

        order.status = 'APPROVED'

        return order

    # -------------------------------------------------------------------------
    def _check_order_type(self, order):

        if not order.type in self.symbol.order_types:
            error = f'order type {order.type} not allowed'
            return self._add_error(order, error)

        return order

    def _check_parameter_type(self, order: Order) -> Order:
        
        if order.last_price is None:
            error = f'missing parameter: last_price'
            return self._add_error(order, error)           
        
        try:
            order.last_price = float(order.last_price)
        except:
            error = f'illegal format for last_price {type(order.last_price)}'
            return self._add_error(order, error)
        
        return order

    def _check_base_or_quote(self, order: Order) -> Order:
        # if we got values for base and quote quantity -> reject order
        if order.base_qty is not None and order.quote_qty is not None:
            error = 'base quantity and quote quantity are mutually exclusive, '
            error += 'but both were specified'
            return self._add_error(order, error)
        
        # check if we have at least one valid value for 'base asset 
        # quantity' or 'quote asset quantity'
        if order.base_qty is not None:
            try:
                order.base_qty = float(order.base_qty)
            except:
                error = f'illegal value for base quantity {order.base_qty}'
                return self._add_error(order, error)
            
        if order.quote_qty is not None:
            try:
                order.quote_qty = float(order.quote_qty)
            except:
                error = f'illegal value for quote quantity {order.quote_qty}'
                return self._add_error(order, error)
            
        # .. if we got neither -> reject order
        if (order.base_qty == 0 or order.base_qty is None) \
            and (order.quote_qty == 0 or order.quote_qty is None):
                    
            error = f'missing values for base _and_ quote quantity '
            error += f'({order.base_qty=} / {order.quote_qty=})'
            return self._add_error(order, error)
            
        return order     
        
    def _check_step_size(self, order):
        
        if order.base_qty is not None:
            
            step_precision = self.symbol.lot_size_step_precision

            if order.side == 'BUY': # and order.type == 'STOP_LOSS_LIMIT':
                order.base_qty = math.ceil(
                    order.base_qty * 10**step_precision) / 10**step_precision
            else: 
                order.base_qty = math.floor(
                    order.base_qty * 10**step_precision) / 10**step_precision

        return order

    def _check_position_size_market_order(self, order: Order):
        
        def _get_quote_from_base(base_qty):
            res = base_qty * order.last_price 
            return math.floor(res * 10**tick_precision) / 10**tick_precision
        
        # .....................................................................
        max_ = order.symbol.market_lot_size_max
        min_ = order.symbol.market_lot_size_min
        min_notional = order.symbol.min_notional
        min_notional_to_market = order.symbol.min_notional_apply_to_market
        tick_precision = order.symbol.tick_precision
        step_precision = order.symbol.lot_size_step_precision

        # .....................................................................
        if not order.base_qty:
            base_qty = order.quote_qty / order.last_price # type:ignore
            base_qty = math.floor(
                base_qty * 10**step_precision) / 10**step_precision
            quote_qty = order.quote_qty
        else:
            base_qty = order.base_qty
            try:
                quote_qty = _get_quote_from_base(order.base_qty)
            except Exception:
                base_qty = order.base_qty
                last_price = order.last_price 
                order.validation_errors.append(
                    f'failed to determine quote qty'\
                        f' from {base_qty} * {last_price}'    
                )
                order.status = 'REJECTED'
                return order
                
        # .....................................................................
        # check if order size exceeds max allowed quantity
        if base_qty > max_: 
            warning = f'{self.name}: order size {base_qty} lowered to {max_}'
            order.validation_warnings.append(warning)
            base_qty = max_
            quote_qty = _get_quote_from_base(base_qty)
        # check if order size is below minimum quantity
        if base_qty < min_:
            order.validation_errors.append(
                f'{self.name}: order size {base_qty:.8f} below minimum: {min_}'
            )
            order.status = 'REJECTED'
        # check if quote asset quantity is below required minimum
        elif quote_qty and quote_qty < min_notional and min_notional_to_market:
            order.validation_errors.append(
                f'{self.name}: notional value ({quote_qty:.8f})'
                    f'lower than min notional: {min_notional}'
            )
            order.status = 'REJECTED'

        # .....................................................................
        # in certain cases (market buy on Kucoin with 'auto borrow') market
        # buy orders that don´t specify the quote asset quantity (=how much
        # do we want to spend) will be rejected by the API.
        # 
        # derive the quote asset quantity from the given base asset quantity
        # based on 'last price'. the actual base asset quantity will then 
        # vary slightly because of slippage, so it is advised that the 
        # caller uses 'quote_qty' by default.
        if order.side == 'BUY' and order.auto_borrow:
            if order.quote_qty is None or order.quote_qty == 0: 
                _get_quote_from_base(base_qty)
                order.quote_qty = quote_qty
                order.base_qty = None
                order.validation_warnings.append(
                    f'For orders with AUTO_BORROW, quote asset quantity '\
                        'should be used instead of base asset quantity'
                )
        else:
            if order.base_qty is not None:
                order.base_qty = base_qty
            if order.quote_qty is not None:
                order.quote_qty = quote_qty
            
        return order

    def _check_position_size_limit_order(self, order: Order) -> Order:
        def _add_error(error):
            order.validation_errors.append(f'{self.name}: ' + error)

        # ......................................................................
        max_ = order.symbol.lot_size_max
        min_ = order.symbol.lot_size_min
        min_n = order.symbol.min_notional
        
        if not order.base_qty:
            _add_error(
                'limit order position check failed -> missing base asset quantity'
            )
            return order
        
        if order.base_qty > max_:           
            order.validation_warnings.append(
                f'{self.name}: order size {order.base_qty} lowered to {max_}'
            )
            order.base_qty = max_  

        if order.base_qty < min_:
            _add_error(
                f'order size {order.base_qty} too small! min size = {min_}'
            )

        if order.base_qty * order.limit_price < min_n:
            _add_error(                
                f'notional value {order.base_qty * order.limit_price} '\
                    f'lower than min notional: {min_n}'
            )

        if order.validation_errors:
            order.status = 'REJECTED'

        return order

    def _check_prices(self, order: Order) -> Order:

        # helper function to append validation errors 
        # and change order status if necessary
        def _add_error(error):
            order.validation_errors.append(f'{self.name}: ' + error)
            order.status = 'REJECTED'

        if not order.last_price:
            _add_error("unable to do price checks: missing 'last price'")
            return order
            
        # ---------------------------------------------------------------------
        min_ = order.symbol.price_filter_min_price 
        max_ = order.symbol.price_filter_max_price
        mult_down = order.last_price * self.symbol.multiplier_down
        mult_up = order.last_price * self.symbol.multiplier_up
        precision = order.symbol.tick_precision

        # check if the 'limit price' is within the boundaries given 
        # by the exchange
        if order.limit_price is None:
             _add_error(f'missing limit price: <None>')
             return order
        
        if order.limit_price < min_:
            _add_error(f'limit price is below the minimum price ({min_:.8f})')

        elif order.limit_price > max_:
            _add_error(f'limit price is above the maximum price ({max_:.8f})')


        if order.limit_price < mult_down:
             _add_error(
                 f'limit price is too far below last price (min: {mult_down:.8f})')

        elif order.limit_price > mult_up:
            _add_error(
                f'limit price is too far above last price (max: {mult_up:.8f})')

        if not order.status == 'REJECTED': 
            # order.limit_price = math.floor(order.limit_price * 10**precision) / 10 ** precision
            order.limit_price = round(order.limit_price, precision)


        # do the same checks for the 'stop price' if the order type has a 'stop price'
        if order.stop_price:

            if order.stop_price < min_:
                _add_error(f'stop price is below the minimum price ({min_})')

            elif order.stop_price > max_:
                _add_error(f'stop price is above the maximum price ({max_})')


            if order.stop_price < mult_down:
                _add_error(f'stop price is too far below current price (min: {mult_down})')

            elif order.stop_price > max_:
                _add_error(f'stop price is too far above current price (max: {mult_up})')            

            if not order.status == 'REJECTED': 
                # order.stop_price = math.floor(order.stop_price * 10**precision) / 10 ** precision
                order.stop_price = round(order.stop_price, precision)

        return order

    def _check_stop_prices(self, order: StopOrder) -> StopOrder:
        
        if not order.last_price:
            order.validation_errors.append(
                "unable to check stop price! missing: 'last price'"
            )
            return order

        if not order.stop_price:
            order.validation_errors.append(
                "unable to check stop price! missing: 'stop price'"
            )
            return order            
        
        # .....................................................................
        if order.side == 'SELL':
            if order.stop_price > order.last_price:
                error = f'stop price ({order.stop_price}) above last price '\
                    f'({order.last_price})! order would trigger immediately'
                return self._add_error(order, error)

        if order.side == 'BUY':
            if order.stop_price < order.last_price:
                error = f'stop price ({order.stop_price}) below last price '\
                    f'({order.last_price})! order would trigger immediately'
                return self._add_error(order, error)

        # .....................................................................
        if isinstance(order, LimitOrder) and order.limit_price:
            if order.side == 'SELL':
                if order.limit_price > order.stop_price:
                    error = f'limit price ({order.limit_price}) above '\
                        f'stop price {order.stop_price}'
                    return self._add_error(order, error)

            if order.side == 'BUY':
                if order.limit_price < order.stop_price:
                    error = f'limit price ({order.limit_price}) below '\
                        f'stop price {order.stop_price}'
                    return self._add_error(order, error)
        
        return order
    
    # --------------------------------------------------------------------------
    def _add_warning(self, order, warning: str, code: int=0):
        try:
            order.validation_warnings.append(
                f'[{self.name}] {code} - {warning}'
            )
        except:
            pass
        
        return order

    def _add_error(self, order, error:str, code: Union[int, str]=0):
        
        code = f'{code} -' if code else ''
        
        try:
            order.validation_warnings.append(f'[{self.name}] {code}{error}')
        except:
            pass
        
        order.status = 'REJECTED'
        return order
# =============================================================================
class Saturn:

    def __init__(self, **kwargs):

        self.symbol = kwargs.get('symbol', None)

        # elf.psm = PositionSizeManager(**kwargs)
        self.cm = ComplianceManager(**kwargs)
        # self.slm = StopLossManager(**kwargs)




# =============================================================================

if __name__ == '__main__':

    pass