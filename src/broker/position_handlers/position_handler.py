#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat August 13  19:07:23 2022

@author: dhaneor
"""
import logging
from typing import Iterable, List, Union, Dict, Tuple

from ..ganesh import Ganesh
from ..models.position import Position
from ..models.symbol import Symbol
from .actions import Action, ActionFactory
from .workers import (PositionWorker, StopLossWorker, TakeProfitWorker, 
                      LoanWorker, DummyWorker, AbstractWorker,
                      ActiveOrders)
from .workers import PositionUpdateRequest , LoanRequest, TakeProfitRequest

logger = logging.getLogger('main.handlers')
logger.setLevel(logging.INFO)

# =============================================================================
"""
TODO    finish implementing the handling of TAKE PROFIT orders 

TODO    implement error handling ... what do we do when a request 
        cannot be executed or returns an error message? abort? 
        rollback of the operation? ...?  

TODO    sometimes __get_active_stop_loss finds SL orders as ACTIVE
        although they were already cancelled. I'm not sure if this a 
        problem with the orders repository or the Kucoin API (maybe
        the latter as such things also happened with the Binance API),
        for now we just use the last order in the list to prevent
        double action here. This is not ideal and means that we can
        only use one STOP LOSS at a time. This should be OK most of 
        the time, but the code is built in a way to allow for multiple
        simultaneous SL orders, so this should be fixed sooner or later! 
"""

# =============================================================================
# TYPE HINT definitions
Worker = Union[AbstractWorker, DummyWorker]



# =============================================================================
class PositionHandlerBuilder:
    
    def __init__(self):

        self.position: Position
        self.action_factory: ActionFactory
        
        logger_name = 'main.handlers.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

    # --------------------------------------------------------------------------
    def build_position_handler(self, position: Position):
        
        if not position.request:
            return
        
        self.position = position.accept_consultant()   
        self.action_factory = ActionFactory(broker=position.broker)
    
        # the positions in the account always include the 'position' 
        # for our currently used quote asset.
        # and here we filter this out as it doesn´t make sense (and
        # would also lead to errors) to process this. The handler
        # is set to None when instantiating a Position so we don´t
        # need to do anything here bc None is valid for 
        # position.handler and is expected as possible value in 
        # later stages.
        if self.position.quote_asset == self.position.balance.asset:
            return
        
        # if there is any Position that doesn't have the 'symbol'
        # attribute set, then we are unable to proceed  
        if not position.symbol:
            raise ValueError(
                f'unable to build position handler! symbol is: {position.symbol}'
            )
        
        # determine Action based on change request and position balance
        try:
            action = self.action_factory.get_action(
                symbol = position.symbol,
                balance=position.balance, 
                request=position.request
            )
        except Exception as e:
            self.logger.exception(
                f'unable to determine action for {position}: {e}'
            )
            action = None
                
        # build the handler
        if action:
            self.position.handler = self.__build_the_handler(action)
            self.position.pending_action = action
            
    # ..........................................................................
    def __build_the_handler(self, action:Action):

        if not action:
            return None
   
        broker = self.position.broker
        
        handler_kwargs = {
            'broker' : self.position.broker,
            'symbol' : self.position.symbol,
            'action' : action
            }
        
        worker_kwargs = {
            'broker' : self.position.broker,
            'symbol' : self.position.symbol
            }
        
        handler = PositionHandler(**handler_kwargs)
        
        if action.position_action:
            handler.position_worker = PositionWorker(**worker_kwargs)
        if action.stop_loss_action:
            handler.stop_loss_worker = StopLossWorker(**worker_kwargs)
            handler.active_stop_loss_orders = \
                broker.get_active_stop_orders(self.position.symbol.name) # type: ignore
        if action.take_profit_action:
            handler.take_profit_worker = TakeProfitWorker(**worker_kwargs)
        if action.repay:
            handler.loan_worker = LoanWorker(**worker_kwargs)
            
        return handler


    def __build_dummy_handler(self):
        return DummyHandler()


# ============================================================================== 
class PositionHandler:
    
    def __init__(self, broker: Ganesh, symbol: Symbol, action: Action):
        
        self.broker: Ganesh = broker
        self.symbol: Symbol = symbol
        self.action: Action = action

        self.take_profit_worker: Worker = DummyWorker()
        self.stop_loss_worker: Worker = DummyWorker()
        self.position_worker: Worker = DummyWorker()
        self.loan_worker: Worker = DummyWorker()
        
        self.active_stop_loss_orders: ActiveOrders = None
        self.active_take_profit_orders: ActiveOrders = None
        
        logger_name = 'main.handlers.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        
        self.proceed: bool = True
  
    def __repr__(self):
        out = f'[{self.__class__.__name__}] {self.position_worker}, '\
            f'{self.stop_loss_worker}, {self.take_profit_worker}'
        
        return out

    # --------------------------------------------------------------------------    
    def execute(self):

        div = '~-•-~' * 10
        self.logger.info(f'{div} updating position for {self.symbol.name} {div}')
        # self.logger.debug(self.broker.get_balance(self.symbol.base_asset))
        # self.logger.debug(self.broker.get_balance(self.symbol.quote_asset))
        self.logger.info(self.action)

        while self.proceed:
            self._cancel_take_profit_orders()
            self._cancel_stop_loss_orders()
            self._update_position()
            self._repay_loan()
            self._create_stop_loss_orders()
            self._create_take_profit_orders()
            break

    # --------------------------------------------------------------------------             
    def _cancel_take_profit_orders(self):
        if self.action.take_profit_current \
            and self.action.take_profit_action in ['cancel', 'keep', 'update']:
            self.logger.info(f'cancelling TP orders ...')
            try:
                self.take_profit_worker.cancel_active_orders()
            except Exception as e:
                self.logger.exception(e)
                self.proceed = False
    
    def _cancel_stop_loss_orders(self):
        if self.action.stop_loss_current and self.action.stop_loss_action:
            try:
                self.stop_loss_worker.cancel_active_orders(
                    self.active_stop_loss_orders
                )
            except Exception as e:
                self.logger.exception(e)
                self.proceed = False
                
    def _update_position(self): 
        if self.action.position_action:
            request = self._build_position_update_request()
            self.logger.debug('updating position size')            
            self.logger.debug(request)
            
            if request:
                try:
                    self.position_worker.execute_request(request)
                except Exception as e:
                    self.logger.exception(e)
                    self.proceed = False
    
    def _repay_loan(self):
        if not self.action.repay:
            return
        
        for idx, condition in enumerate(self.action.repay):        
            
            if condition:                      
                asset = self.symbol.base_asset if idx == 0 \
                    else self.symbol.quote_asset 
                
                repay_amount = self._get_repay_amount(asset)
                
                if repay_amount > 0:   
                    request = LoanRequest(
                        action='repay',
                        asset=asset,
                        amount=repay_amount
                    )
                    
                    try:
                        self.loan_worker.execute_request(request)
                    except Exception as e:
                        self.logger.exception(e)
                        self.proceed = False
                    
                else:    
                    self.logger.debug(
                        f'Although repay_action was set to <{condition}> '\
                            f'for {asset}, borrowed amount was: {repay_amount}'
                    )
                
            if idx == 1:
                return
    
    def _create_stop_loss_orders(self):  
        if self.action.stop_loss_action and self.action.stop_loss_target:      
            sl_requests = self.action.stop_loss_target
            
            if not self.action.position_action:
                net = self.action.balance_target
            else:
                balance = self.broker.get_balance(self.symbol.base_asset)
                net = balance['free'] - balance['borrowed']
            
            amounts = self._get_order_amounts(net, [r[1] for r in sl_requests])
            
            request = tuple( 
                {
                    'side' : 'SELL' if net > 0 else 'BUY',
                    'type' : 'market', 
                    'stop price': req[0], 
                    'base_qty' : amounts[idx], 
                    'percent' : req[1]
                } \
                for idx, req in enumerate(sl_requests)
            )
            
            try:
                self.stop_loss_worker.execute_request(request)
            except Exception as e:
                self.logger.exception(e)
                self.proceed = False
 
    def _create_take_profit_orders(self):
        if self.action.take_profit_target \
            and self.action.take_profit_action in ['create', 'update']:

            tp_requests = self.action.take_profit_target
            
            balance = self.broker.get_balance(self.symbol.base_asset)
            net = balance['free'] - balance['borrowed']
            
            percentages = [item[1] for item in tp_requests]
            amounts = self._get_order_amounts(net, percentages)
            
            self.logger.debug(f'{balance=}')
            self.logger.debug(f'amounts')
            
            side = 'SELL' if net > 0 else 'BUY'
            
            # fields = ['side', 'asset', 'base_qty', 'quote_qty', 'limit_price', 'type']
            request = tuple(TakeProfitRequest( 
                side=side, 
                base_qty=amounts[idx], 
                limit_price=req[0], 
                type='limit'
                ) \
                for idx, req in enumerate(tp_requests))

            try:
                self.take_profit_worker.execute_request(request)
            except Exception as e:
                self.logger.exception(e)
                self.proceed = False
    
    # --------------------------------------------------------------------------
    def _build_position_update_request(self):

        change = self.action.balance_change

        if change > self.symbol.lot_size_min:
            side = 'BUY'
        elif change < 0 - self.symbol.lot_size_min:
            side = 'SELL'
        else:
            self.logger.warning(
                'invalid action: position change < minimum lot size'
            )
            return None

        # .....................................................................
        request = PositionUpdateRequest(
            side=side, asset=self.symbol.base_asset, base_qty=change, 
            quote_qty=None, type='market' 
        )
        
        return request
    
    def _get_order_amounts(self, asset_balance:float, request:List[float]
                           ) -> List[float]:
        """Get order amounts based on percentages and a gross amount.
        
        NOTE:   If the sum of the percentages exceeds 1, every single
                amount in the resulting list will be scaled down 
                proportionally! A warning will be issued, but we 
                don't raise an exception

        :param asset_balance: the amount that should be covered by the
        resulting orders
        :type asset_balance: float
        :param request: a list that includes percentages (0..1) to use
        :type request: List[float]
        :return: a list with the concrete order amounts, summing to 1 
        :rtype: List[float]
        """
        amounts = [asset_balance * percentage for percentage in request]
        
        if sum(amounts) > asset_balance:
            factor = sum(amounts) / asset_balance 
            amounts = [item / factor for item in amounts] 
            self.logger.warning(
                f'get_order_amounts had to scale down the percentages'
            )
            
        return amounts
     
    def _get_repay_amount(self, asset) -> float:
        balance = self.broker.get_balance(asset)
        self.logger.debug(balance)
        
        free, amount = balance['free'], balance['borrowed']
        return amount if amount < free else free


class DummyHandler:
    
    def __init__(self):
        self.action = None 
        
    def __repr__(self):
        return '[DummyHandler]' 
    
    def execute(self):
        pass   
    
