#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun August 14  19:47:23 2022

@author: dhaneor
"""
import concurrent.futures
import logging

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Iterable, List, Tuple, Union, Callable, NamedTuple

from broker.ganesh import Ganesh
from broker.models.orders import Order
from broker.util.order_factory import OrderFactory
from broker.models.exchange_order import ExchangeOrder, build_exchange_order
from broker.models.symbol import Symbol

logger = logging.getLogger('main.workers')
logger.setLevel(logging.DEBUG)



# ==============================================================================
""" 
TODO    change the order canceller (and also the method in Kucoin class)
        so that we don't fetch the order status after cancelling. this 
        is not necessary and can be made faster by not making a second 
        API call 
"""

# ==============================================================================



fields = ['side', 'asset', 'base_qty', 'quote_qty', 'limit_price', 'type']
defaults = (None, None, None, None, None, 'market')

PositionUpdateRequest = namedtuple(
    'PositionUpdateRequest', field_names=fields, defaults=defaults
    )

# .............................................................................
defaults = (None, None, None, None, None, 'limit')

TakeProfitRequest = namedtuple(
    'TakeProfitRequest', field_names=fields, defaults=defaults
    )

# ............................................................................. 
LoanRequest = namedtuple(
    'LoanRequest', [
        'action', 'asset', 'amount'
        ]
    )

ActiveOrders = Union[Iterable[ExchangeOrder], None]




# ==============================================================================
class AbstractOrderExecutor(ABC):
    
    def __init__(self, broker: Ganesh, symbol: Symbol, 
                 register_errors_to: Callable[[tuple], None]):
        
        self.broker = broker
        self.symbol = symbol
        
        self.failed_requests: Tuple[dict]
        self.notify_about_failed_requests = register_errors_to
        
        logger_name = 'main.workers.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
            
    @property
    def all_went_well(self):
        return True if not self.failed_requests else False
            
    @abstractmethod
    def execute(self):
        pass

    # --------------------------------------------------------------------------
    def _extract_errors_from_order_results(self, order_results:List[dict]):

        return
        self.failed_requests = tuple(
            cr for cr in order_results if not o['status'] == 200
            )
        
        [self.notify_about_failed_requests(fr) for fr in self.failed_requests]
 
    
class OrderCanceller(AbstractOrderExecutor):
    
    def __init__(self, broker:Ganesh, symbol:Symbol, 
                 register_errors_to: Callable):
        super().__init__(broker, symbol, register_errors_to)
        self.cancelled_order_ids: tuple

    def execute(self, orders:Iterable[ExchangeOrder]):
        order_ids_to_cancel = self.__extract_order_ids(orders)

        if order_ids_to_cancel:
            self.__cancel__orders_by_id(order_ids_to_cancel)
            
        for o in orders:
            if o.order_id in self.cancelled_order_ids:
                o.status = 'CANCELED'
            else:
                o.status = 'CANCEL FAILED'
            
            self.logger.info(o)
    
    # --------------------------------------------------------------------------
    def __extract_order_ids(self, orders: Iterable[ExchangeOrder]) -> Tuple[str]:
        return tuple((o.order_id for o in orders)) 
            
    def __cancel__orders_by_id(self, order_ids:tuple):
        cancel_results = self.__concurrently_cancel_orders(order_ids)

        # TODO remember the cancelled order ids, so we can restore the
        # orders if necessary
        
        for cr in cancel_results:
            if cr.get('success'):
                message = cr['message']['cancelledOrderIds']
                self.cancelled_order_ids = tuple(oid for oid in message)
            else:
                error, error_code = cr['error'], cr['error code'] 
                self.logger.error(
                    f'cancelling order failed: {error} ({error_code})'
                )
            
        # self._extract_errors_from_order_results(cancel_results)

    def __concurrently_cancel_orders(self, order_ids:tuple) -> List[dict]:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            
            _futures = []
            for oid in order_ids:
                _futures.append(
                    executor.submit(self.broker.cancel, 
                                    order_id=oid,
                                    detailed_result=False)
                )

            return [
                future.result() for future \
                    in concurrent.futures.as_completed(_futures)
                ]
 
 
class OrderCreator(AbstractOrderExecutor):
    
    def __init__(self, broker: Ganesh, symbol: Symbol, 
                 register_errors_to: Callable):
        
        super().__init__(broker, symbol, register_errors_to)
        self.created_order_ids: tuple

    def execute(self, orders: Iterable[Order]):
        
        orders = tuple(o for o in orders if isinstance(o, Order))
        
        if orders:
            self.__execute_orders(orders)
        else:
            raise ValueError(
                f"Please provide an Iterable with instances of 'Order'!"
            )

    # --------------------------------------------------------------------------
    def __execute_orders(self, orders: Tuple[Order]):
        order_results = self.__concurrently_execute_orders(orders)        
        [self.logger.info(o) for o in order_results]
            
        # self._extract_errors_from_order_results(order_results)
    
    def __concurrently_execute_orders(self, orders:Tuple[Order]) -> List[dict]:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            
            _futures = []
            for order in orders:
                order.auto_borrow=True
                
                _futures.append(executor.submit(
                    self.broker.execute, 
                    order=order,
                    detailed_result=False
                    )
                )

            return [
                future.result() for future \
                    in concurrent.futures.as_completed(_futures)
                ]



# ==============================================================================
class AbstractWorker(ABC):
    
    def __init__(self, broker:Ganesh, symbol:Symbol):
        
        super().__init__()
        
        self.name = None
        self.broker: Ganesh = broker
        self.symbol: Symbol = symbol

        self.order_factory = OrderFactory()
        
        self.order_creator = OrderCreator(
            broker=self.broker, symbol=self.symbol, 
            register_errors_to=self._register_failed_requests
            )      
        self.order_canceller = OrderCanceller(
            broker=self.broker, symbol=self.symbol, 
            register_errors_to=self._register_failed_requests
            )
        
        self.failed_requests: list = []
        
        logger_name = 'main.workers.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        
    def __repr__(self):
        return self.__class__.__name__
    
    @property
    def last_traded_price_for_symbol(self) -> float: 
        return self.broker.get_last_price(self.symbol.name)
    
    # -------------------------------------------------------------------------- 
    @abstractmethod
    def execute_request(self, request):
        pass
    
    def cancel_active_orders(self, active_orders: ActiveOrders=None) -> bool:

        orders_to_cancel = \
            active_orders if active_orders else self._get_active_orders()
            
        if not orders_to_cancel:
            self.logger.warning('found no orders to cancel!')
                
        self.order_canceller.execute(orders_to_cancel)

        return False if self.failed_requests else True

    def create_orders(self, orders: Union[Order, Tuple[Order]]):
        if not isinstance(orders, tuple):
            orders = (orders,)
        
        valid_orders = tuple(
            o for o in orders \
                if o.type in self.__get_valid_order_types()\
                    and o.status == 'APPROVED'
        )
        
        [self.logger.warning(o) for o in orders if o not in valid_orders]
        
        if valid_orders:
            self.order_creator.execute(valid_orders)

    def restore_orders(self):
        order_ids = self.order_canceller.cancelled_order_ids
        
        if order_ids:
            orders = self.__recreate_orders_from_order_id(order_ids)
            self.create_orders(orders)
        
    # ------------------------------------------------------------------------------
    def _register_failed_requests(self, response_for_failed_request: dict):
        self.failed_requests.append(response_for_failed_request)
    
    def _get_active_orders(self):
        raise NotImplementedError(f'_get_active_orders() is not implemented')
    
# -----------------------------------------------------------------------------
    def __get_valid_order_types(self):
        return self.symbol.order_types

    def __recreate_orders_from_order_id(self, order_ids:Union[str, tuple]
                                        ) -> Tuple[Order]:
        if isinstance(order_ids, str):
            order_ids = tuple(order_ids)

        return tuple(
            self.__build_order_from_order_id(oid) for oid in order_ids
            )

    # @abstractmethod
    def __build_order_from_order_id(self, order_id) -> Order: # type: ignore
        pass

    # @abstractmethod
    def __build_order_from_request(self):
        pass
 
    
class StopLossWorker(AbstractWorker):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'STOP LOSS Worker'
    
    def execute_request(self, request: Tuple[dict]):
        self.logger.debug(f'received request: {request}')
        orders = tuple(
            self.__build_order_from_request(sl_req) for sl_req in request     
        )
        
        self.create_orders(orders)
    
    def cancel_active_orders(self, orders: ActiveOrders=None) -> bool:
        return super().cancel_active_orders(orders)
    
    def create_orders(self, orders: Tuple[Order]):
        return super().create_orders(orders)
    
    def restore_orders(self):
        return super().restore_orders()
      
    # --------------------------------------------------------------------------    
    def _get_active_orders(self):
        return self.broker.get_active_stop_orders(self.symbol.name)[-1:]
    
    # --------------------------------------------------------------------------
    def __get_valid_order_types(self):
        return tuple(ot for ot in self.symbol.order_types if 'STOP' in ot)

    def __build_order_from_order_id(self, order_id) -> Order:
        previous_order = self.broker.get_order(order_id)
        
        if not previous_order:
            raise ValueError(
                f'cannot build order! order id {order_id} is unknown'
            )
        
        # the stop loss order to be restored can be a STOP_LOSS_MARKET
        # order or a STOP_LOSS_LIMIT_ORDER and only the latter has a
        # field/value 'limit price'
        try:
            limit_price = previous_order.price
        except:
            limit_price = None
        
        if previous_order.side == 'SELL':    
            build_function = self.order_factory.build_long_stop_order
        else:
            build_function = self.order_factory.build_short_stop_order
            
        return build_function(
            symbol=self.symbol,
            type=previous_order.type,
            base_qty=previous_order.orig_qty,
            stop_price = previous_order.stop_price,
            limit_price=limit_price,
            last_price=self.last_traded_price_for_symbol
        )        
 
    def __build_order_from_request(self, request: dict):
        
        if request['side'] == 'SELL': 
            build_function = self.order_factory.build_long_stop_order
        else:
            build_function = self.order_factory.build_short_stop_order
        
        return build_function(
            symbol=self.symbol,
            type=request.get('type', ''),
            base_qty=abs(request.get('base_qty', 0)),
            stop_price=request.get('stop price', -1),
            last_price=self.last_traded_price_for_symbol
            )

            
class TakeProfitWorker(AbstractWorker):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TAKE PROFIT Worker'
    
    def execute_request(self, request:Tuple[dict]):
        self.logger.debug(f'received request: {request}')
        orders = tuple(
            self.__build_order_from_request(sl_req) for sl_req in request     
        )
        
        self.create_orders(orders)
    
    def cancel_active_orders(self) -> bool:
        return super().cancel_active_orders()
    
    def create_orders(self, orders: Tuple[Order]):
        return super().create_orders(orders)
    
    def restore_orders(self):
        return super().restore_orders()
      
    # --------------------------------------------------------------------------    
    def _get_active_orders(self):
        return [
            o for o in self.broker.orders \
                if all(arg for arg in (o.symbol == self.symbol.name,
                                       o.status == 'NEW',
                                       o.type == 'LIMIT'
                                       )
                       )
            ]
    
    # --------------------------------------------------------------------------
    def __get_valid_order_types(self):
        return tuple(ot for ot in self.symbol.order_types if 'LIMIT' in ot)

    def __build_order_from_order_id(self, order_id) -> Order:
        previous_order = self.broker.get_order(order_id)
        
        if not previous_order:
            raise ValueError(
                f'cannot build order! order id {order_id} is unknown'
            )
            
        
        if previous_order.side== 'SELL': 
            build_function = self.order_factory.build_sell_order
        else:
            build_function = self.order_factory.build_buy_order
        
        return build_function(
            symbol=self.symbol,
            type=previous_order.type,
            base_qty=previous_order.orig_qty,
            limit_price=previous_order.price,
            last_price=self.last_traded_price_for_symbol
            )

    def __build_order_from_request(self, request):
        
        if request.side == 'SELL': 
            build_function = self.order_factory.build_sell_order
        elif request.side == 'BUY':
            build_function = self.order_factory.build_buy_order
        else:
            raise ValueError(
                f'invalid side in PositionUpdateRequest: {request.side}'
            )
        
        return build_function(
            symbol=self.symbol,
            type=request.type,
            base_qty=abs(request.base_qty),
            quote_qty=request.quote_qty,
            limit_price=request.limit_price,
            last_price=self.last_traded_price_for_symbol
            )
  
    
class PositionWorker(AbstractWorker):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'POSITION Worker'
    
    # --------------------------------------------------------------------------  
    def execute_request(self, request: PositionUpdateRequest):
        self.logger.debug(f'received request: {request}')
        order = self.__build_order_from_request(request)
        self.logger.info(order)
        self.create_orders(order)
    
    def cancel_active_orders(self) -> bool:
        return super().cancel_active_orders()
    
    def create_orders(self, orders: Union[Order, Tuple[Order]]):
        return super().create_orders(orders) 
    
    def restore_orders(self):
        pass
    
    # --------------------------------------------------------------------------    
    def _get_active_orders(self) -> List[ExchangeOrder]:
        return [
            o for o in self.broker.orders \
                if all(
                    arg for arg in (
                        o.symbol == self.symbol.name,
                        o.status == 'NEW',
                        )
                    )
                ]
    
    # --------------------------------------------------------------------------
    def __get_valid_order_types(self) -> tuple:
        return tuple(ot for ot in self.symbol.order_types if not 'STOP' in ot)

    def __build_order_from_order_id(self, order_id) -> Order: # type: ignore
        pass

    def __build_order_from_request(self, request: PositionUpdateRequest) -> Order:
        
        if request.side == 'SELL': 
            build_function = self.order_factory.build_sell_order
        elif request.side == 'BUY':
            build_function = self.order_factory.build_buy_order
        else:
            raise ValueError(
                f'invalid side in PositionUpdateRequest: {request.side}'
            )
        
        return build_function(
            symbol=self.symbol,
            type=request.type,
            base_qty=abs(request.base_qty),
            quote_qty=request.quote_qty,
            limit_price=request.limit_price,
            last_price=self.last_traded_price_for_symbol
            )
    

class LoanWorker(AbstractWorker):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'LOAN Worker'
        
    def execute_request(self, request:LoanRequest):
        
        self.logger.info(f'received {request}')
        
        if request.action == 'borrow':
            res = self._borrow(asset=request.asset, amount=request.amount)
            
            if res.get('success'):
                self.logger.info(
                    f'successfully borrowed {request.amount} {request.asset}'
                    )
            else:
                error_code = res.get('error code')
                error = res.get('error')
                amount, asset = request.amount, request.asset
                error_str = f'unable to borrow {amount} {asset} '
                error_str += f'({error_code}: {error})'
                self.logger.critical(error_str)
        
        elif request.action == 'repay':
            res = self._repay(asset=request.asset, amount=request.amount) 

            if res.get('success'):
                self.logger.info(
                    f'successfully repaid {request.amount} {request.asset}'
                    )
            else:
                error_code = res.get('error code')
                error = res.get('error')
                amount, asset = request.amount, request.asset
                error_str = f'unable to repay {amount} {asset} '
                error_str += f'({error_code}: {error})'
                self.logger.critical(error_str)
        
        else:
            raise ValueError('unkown action: {request.action}')
        
        return res
    
    # -------------------------------------------------------------------------         
    def _borrow(self, asset:str, amount:float):
        return self.broker.borrow(asset=asset, amount=amount)
    
    def _repay(self, asset:str, amount:float):
        return self.broker.repay(asset=asset, amount=amount)
        

        
class DummyWorker:
    
    def __init(self):
        pass
    
    def __repr__(self):
        return 'DummyWorker'
    
    def execute_request(self, *args, **kwargs):
        pass
    
    def cancel_active_orders(self, active_orders: ActiveOrders = None) -> bool:
        return True
    
    def restore_orders(self):
        pass
    
    def _get_active_orders(self):
        pass
    
    