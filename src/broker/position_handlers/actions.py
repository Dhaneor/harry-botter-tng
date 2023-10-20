#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:22:20 2022

@author dhaneor
"""
import logging
from dataclasses import dataclass
from copy import deepcopy
from typing import Iterable, List, Union, Dict, Tuple

from ..ganesh import Ganesh
from ..models.symbol import Symbol 
from ..models.requests import PositionChangeRequest
from ..models.balance import Balance

# =============================================================================
@dataclass
class Action:
    symbol: str = ''
    
    position_current: Union[str, None] = None
    position_target: Union[str, None] = None
    position_action: Union[str, None] = None
    position_change: Union[str, None] = None
    close: bool = False
    open: bool = False
    update: bool = False

    balance_current: float = 0.00
    balance_target: float = 0.00
    balance_change: float = 0.00

    stop_loss_current: Union[Iterable[tuple], None] = None
    stop_loss_target: Union[Iterable[tuple], None] = None
    stop_loss_action: Union[str, None] = ''
    
    take_profit_current: Union[Iterable[tuple], None] = None
    take_profit_target: Union[Iterable[tuple], None] = None
    take_profit_action: Union[str, None] = None
    
    repay: Union[List[bool], None] = None


class ActionFactory:
    
    def __init__(self, broker: Ganesh):
        self.symbol: str
        self.symbol_obj: Union[Symbol, None]
        self.balance: Balance
        self.request: PositionChangeRequest
        self.broker: Ganesh = broker

        logger_name = 'main.handlers.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
    def get_action(self, symbol: Symbol, balance: Balance, 
                   request: PositionChangeRequest) -> Union[Action, None]:
        
        self.balance, self.request = balance, request
        self.symbol = symbol.name
        self.symbol_obj = symbol
        
        if not self.symbol_obj:
            raise ValueError(
                f'unable to get symbol information for {self.symbol}'
            )
        
        action = Action()

        action.symbol = self.symbol        
        action = self._add_current_state(action)
        action = self._add_target_state(action)
        action = self._calculate_balance_change(action)
        
        actions = self._determine_actions(action)
        
        if actions:
            self.logger.debug(actions)
                
        return actions
    
    # -------------------------------------------------------------------------
    def _add_current_state(self, action: Action) -> Action:
        if self.balance.net != 0:
            self.logger.debug(f'{self.balance.asset}: {self.balance.net}')

        action.position_current = self.__get_position_type(self.balance.net)
        action.balance_current = self.balance.net
        
        if action.position_current:
            action.stop_loss_current = self.__get_active_stop_loss()
            action.take_profit_current = self.__get_active_take_profit()
    
        return action
    
    def _add_target_state(self, action: Action) ->Action:
        
        action.position_target = self.__get_position_type(self.request.target)
        
        if action.position_target:
            action.balance_target = self.request.target
            
            if self.request.stop_loss:
                self.logger.debug(self.request.stop_loss)
                action.stop_loss_target = self.__sort_sl_or_tp_tuple(
                    tuple(self.request.stop_loss)
                )
            
            if self.request.take_profit:
                action.take_profit_target = self.__sort_sl_or_tp_tuple(
                    tuple(self.request.take_profit)
                )

                
        else:
            action.balance_target = 0
            action.stop_loss_target = None
            action.take_profit_target = None
        
        return action
    
    def _calculate_balance_change(self, action: Action) -> Action:
        action.balance_change = action.balance_target - action.balance_current

        return action
    
    # .........................................................................
    def _determine_actions(self, action) -> Union[Action, None]:
        action = self.__determine_position_action(action)
        action = self._split_action(action)
        action = self.__determine_repay_action(action)
        
        action = self.__determine_stop_loss_action(action)
        action = self.__determine_take_profit_action(action)
        action = self.__coordinate_tp_and_sl_actions(action)
        
        if all(
            arg is None for arg in (
                action.position_action, 
                action.stop_loss_action, 
                action.take_profit_action
            )
        ):
            action = None
                                
        return action
     
    def _split_action(self, action:Action) -> Action:

        if action.position_action:
            if self.__is_a_position_switch(action):
                self.logger.debug('switch position found: splitting action!')
                action = self.__get_close_position_action(deepcopy(action))

            self.logger.debug(action)
        return action
 
    # -------------------------------------------------------------------------
    def __get_position_type(self, asset_balance:float) -> Union[None, str]:
        
        if asset_balance > 0 \
            and not self.__is_dust_or_zero(asset_balance, self.symbol):            
            return 'LONG' 
       
        elif asset_balance < 0 \
            and not self.__is_dust_or_zero(asset_balance, self.symbol):
            return 'SHORT'
        
        else:
            return None
    
    def __get_active_stop_loss(self) -> Union[Tuple[tuple], None]:

        sl_orders = self.broker.get_active_stop_orders(self.symbol) 
        
        if not sl_orders:
            return None
        else:
            order = sl_orders[-1] # see TODO above!
            return ((order.stop_price, 1),)
               
        # gross_base_qty = sum(tuple([o.orig_qty for o in sl_orders]))

        # res = tuple(
        #     (o.stop_price, round(o.orig_qty / gross_base_qty, 2))\
        #     for o in sl_orders
        # )

        # return self.__sort_sl_or_tp_tuple(res)

    def __get_active_take_profit(self) -> Union[Tuple[tuple], None]:
  
        orders = self.broker.get_all_active_orders(self.symbol)
        tp_orders = [o for o in orders if o.type == 'LIMIT']  
        
        if not tp_orders:
            return None  
        
        gross_base_qty = sum(tuple([o.orig_qty for o in tp_orders]))
        
        res = tuple(
            (o.price, round(o.orig_qty / gross_base_qty, 2))\
            for o in tp_orders
        )

        return self.__sort_sl_or_tp_tuple(res)

    def __determine_position_action(self, action:Action) -> Action:

        balance_change_is_dust_or_zero = self.__is_dust_or_zero(
            action.balance_change, action.symbol
        )
        
        balance_target_is_dust_or_zero = self.__is_dust_or_zero(
            action.balance_target, action.symbol
        )
        
        balance_current_is_dust_or_zero = self.__is_dust_or_zero(
            action.balance_current, action.symbol
        )
        
        # .....................................................................
        # DO NOTHING if the requested change in balance (position size)
        # is too small to be exeuted or to matter
        if  balance_change_is_dust_or_zero:
            action.position_action = None
            action.balance_change = 0
        
        # condition that requires to CLOSE the position 
        elif not balance_change_is_dust_or_zero \
            and balance_target_is_dust_or_zero:
            self.logger.debug('balance target is dust or zero -> CLOSE')
            action.close = True
            action.position_action = 'close'
            action.position_change = 'decrease'
            action.stop_loss_action = None
            action.take_profit_action = None
        
        # condition that requires to OPEN a position                
        elif balance_current_is_dust_or_zero\
            and not balance_change_is_dust_or_zero \
                and not balance_target_is_dust_or_zero:
            self.logger.debug('balance target > dust or zero -> OPEN')
            action.position_action = 'open'
            action.position_change = 'increase'
            action.close = False
            action.open = True

        # condition that requires to UPDATE (=increase or decrease size)
        # the position
        elif not balance_current_is_dust_or_zero\
            and not balance_change_is_dust_or_zero \
                and not balance_target_is_dust_or_zero:
            self.logger.debug('balance target > dust or zero -> UPDATE')
            action.update = True
            action.position_action = 'update'
            
            if abs(action.balance_current) > abs(action.balance_target):
                action.position_change = 'decrease'
            elif abs(action.balance_current) < abs(action.balance_target):
                action.position_change = 'increase'
                
            
        # if none of the conditions above is met then either the request 
        # is invalid or we have an error in the logic above
        else:
            raise Exception('unable to determine position change action!')

        return action
  
    def __determine_repay_action(self, action:Action) -> Action:

        # check if there are outstanding loans (for instance because
        # our stop loss or ake profit got triggered)
        action = self.__get_repay_action(action)

        if not action.position_current:
            return action
        
        if action.position_action in (None, 'open'):
            return action

        # .....................................................................
        # check if we need to repay something after reducing a LONG position
        if action.position_current == 'LONG' \
            and action.balance_change < 0:
            
            balance = self.broker.get_balance(self.symbol_obj.quote_asset) # type: ignore
            
            if balance.get('borrowed') != 0:
                
                if not action.repay:
                    action.repay = [False, True]
                else:
                    action.repay[1] = True

                self.logger.debug(f'repay quote: {action.repay[1]} - {balance}')
        
        # check if we need to repay something after reducing a LONG position
        if action.position_current == 'SHORT' \
            and action.balance_change > 0:
            
            balance = self.broker.get_balance(self.symbol_obj.base_asset) # type: ignore
            
            if balance.get('borrowed') != 0:

                if not action.repay:
                    action.repay = [True, False]
                else:
                    action.repay[0] = True
                
                self.logger.debug(f'repay base: {action.repay[0]} - {balance}')
        
        return action
            
    def __determine_stop_loss_action(self, action:Action) -> Action:
        
        if action.stop_loss_current == action.stop_loss_target:
            
            if not action.position_action:
                action.stop_loss_action = None
            else:
                action.stop_loss_action = 'update'
        
        else:

            if action.stop_loss_current and not action.stop_loss_target:
                action.stop_loss_action = 'cancel'
            elif not action.stop_loss_current and action.stop_loss_target:
                    action.stop_loss_action = 'create'
            elif action.stop_loss_current and action.stop_loss_target:
                    action.stop_loss_action = 'update'
            else:
                self.logger.error('-'*200)
                self.logger.error(action)
                self.logger.error('unable to determine stop loss action')
                    
        return action
    
    def __determine_take_profit_action(self, action:Action) -> Action:
            
        if not action.take_profit_target \
            and action.take_profit_current:
                action.take_profit_action = 'cancel'
        
        elif  action.take_profit_target \
            and not action.take_profit_current:
                action.take_profit_action = 'create'

        elif  action.take_profit_target \
            and action.take_profit_current:
        
            if action.take_profit_current == action.take_profit_target \
                and not action.position_action:
                action.take_profit_action = None

            elif action.take_profit_current == action.take_profit_target \
                and action.position_action:
                action.take_profit_action = 'update'
                
            else:
                action.take_profit_action = 'update'
        
        return action

    def __coordinate_tp_and_sl_actions(self, action:Action) -> Action:
        # if there is an active take profit and we want to keep it like
        # it is, we have to temporarily delete and re-create it,
        # if we want to update the stop loss (this is just how it 
        # works in Kucoin) 
        if action.stop_loss_action == 'update' \
            and action.take_profit_target\
                and not action.take_profit_action:
            action.take_profit_action = 'keep'
        
        return action

    # -------------------------------------------------------------------------
    # low level helper methods
    def __get_repay_action(self, action: Action) -> Action:
        base = self.__needs_repayment(self.symbol_obj.base_asset) # type:ignore
        quote = self.__needs_repayment(self.symbol_obj.quote_asset) # type:ignore
        
        if not (base or quote):
            return action

        action.repay = [base, quote]            
        return action
    
    def __needs_repayment(self, asset: str) -> bool:
        balance = self.broker.get_balance(asset)
        return True if balance['free'] and balance['borrowed'] else False
    
    def __sort_sl_or_tp_tuple(self, item: Tuple[tuple]) -> tuple:
        return tuple(sorted(item, key=lambda x: x[0]))
    
    def __get_close_position_action(self, action:Action) -> Action:
        close_action = action
        close_action.balance_target = 0
        close_action.balance_change = -1 * action.balance_current
        close_action.position_target = None
        close_action.position_action = 'close'
        close_action.position_change = 'decrease'
        close_action.close = True
        close_action.open = False
        close_action.update = False
        close_action.stop_loss_target = None
        close_action.take_profit_target = None
        close_action = self.__determine_stop_loss_action(close_action)
        close_action = self.__determine_take_profit_action(close_action)
        close_action = self.__determine_repay_action(close_action)

        return action
    
    def __is_dust_or_zero(self, amount, symbol_name) -> bool:

        return True if abs(amount) < self.symbol_obj.lot_size_min else False # type: ignore

    def __is_a_position_switch(self, action:Action) -> bool:
        
        if action.position_current == 'LONG' \
            and action.position_target == 'SHORT':
            is_a_switch = True
        
        elif action.position_current == 'SHORT' \
            and action.position_target == 'LONG':
            is_a_switch = True
        
        else:
            is_a_switch = False
        
        return is_a_switch
            
