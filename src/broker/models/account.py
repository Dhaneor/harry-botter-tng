#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import logging

from typing import Iterable, List, Tuple, Union
from datetime import datetime

from ..ganesh import Ganesh
from ..position_handlers.position_handler import PositionHandlerBuilder
from ..models.position import Position, PositionFactory
from ..models.requests import RequestFactory
     
# =============================================================================
class Account:
    
    def __init__(self, broker: Ganesh, quote_asset: str):
        self.broker = broker
        
        self.position_factory = PositionFactory(self.broker) 
        self.request_factory = RequestFactory(quote_asset=quote_asset)
        self.handler_builder = PositionHandlerBuilder()
        
        self.quote_asset: str = quote_asset
        
        self.positions: Tuple[Position]
        self.initialize_from_exchange_api()
                
        self.logger = logging.getLogger('main.Account')
        
    def __repr__(self):
        
        return f'[Account] value: {self.get_account_value()} \t'\
            f'account leverage: {round(self.account_leverage, 2)} \t'\
                f'debt ratio: {round(self.debt_ratio, 2)}'
     
     # ------------------------------------------------------------------------  
 
    @property
    def debt_ratio(self):
        return self.broker.debt_ratio
    
    @property
    def account_leverage(self):
        return round((self.debt_ratio / (1-self.debt_ratio) + 1), 2)
        
    # -------------------------------------------------------------------------
    def initialize_from_exchange_api(self):
        self.positions = self.position_factory.build_positions(
            quote_asset=self.quote_asset
        )
   
    def get_all_positions(self, include_dust: bool=True, update: bool=False
                          ) -> Tuple[Position]:
        if update:
            self.initialize_from_exchange_api()
        
        if include_dust:
            return self.positions
        else:
            return tuple(pos for pos in self.positions if not pos.is_dust)
        
    def get_position(self, asset:str):
        return next(filter(lambda x: x.asset == asset, self.positions), None)

    def get_account_value(self, quote_asset: str='') -> float:
        _precision = self.__get_asset_precision(self.quote_asset)
        
        value = sum(
            tuple(
                pos.get_value(quote_asset=self.quote_asset) \
                    for pos in self.positions
                )
        )

        return round(value, _precision)
     
    def reset_account(self, dont_touch: Union[Iterable[str], None]=None):
        self.add_requests_to_positions(target_account=[])                  
        self.add_handler_to_positions()   
        
        if dont_touch:
            for pos in self.positions:
                if pos.asset in dont_touch:
                    pos.handler = None
                    print(pos.asset, pos.handler)        

        self._decrease_positions()
        
    def get_summary(self) -> str:
        
        positions = ''
        for pos in self.positions:
            if not pos.is_dust:
                size = round(pos.balance.net, pos.asset_precision)
                positions += \
                    f'\n[{pos.TYPE}] {size} {pos.asset}'
    
        if self.quote_asset in ('BTC', 'ETH'):
            quote_asset_precision = 6
        else:
            quote_asset_precision = 2
        
        value = round(self.get_account_value(), quote_asset_precision)
        date = datetime.utcnow().replace(microsecond=0)
        div = '-' * 28
        
        summary = f'Account value: {value} {self.quote_asset}'

        return f'{div}\nSTATE OF THE NATION\n@ {date}:\n'\
            f'{positions}\n{div}\n{summary}'
    
    # -------------------------------------------------------------------------
    def add_requests_to_positions(self, target_account:List[dict]):
        """Adds a PositionChangeRequest to all positions in this Account.

        :param target_account: a list of target requests, one for each
        asset that should be changed in one or more ways
        :type target_account: List[dict]
        """
        self.request_factory.prepare_requests(
            target_account=target_account, all_assets=self.broker.valid_assets
        )  
        
        for pos in self.positions:
            pos.add_request(
                self.request_factory.get_request_for_asset(pos.balance.asset)
            )
    
    def add_handler_to_positions(self):
        for position in self.positions:
            if position.request:
                self.handler_builder.build_position_handler(position)
   
    def execute_changes(self, target_account: List[dict]):
        self.add_requests_to_positions(target_account=target_account)
        
        self.add_handler_to_positions()
        self._decrease_positions()
        
        self.add_handler_to_positions()
        self._increase_positions()
        self._update_stop_orders()
   
    def _decrease_positions(self):
        for pos in self.positions:
            if pos.handler and pos.pending_action:
                if pos.pending_action.position_change == 'decrease':
                    pos.handler.execute() # type: ignore
                    pos.handler, pos.pending_action = None, None

    def _increase_positions(self):
        for pos in self.positions:
            if pos.handler and pos.pending_action:
                if pos.pending_action.position_change == 'increase':
                    pos.handler.execute() # type: ignore
                    pos.handler, pos.pending_action = None, None 
                    
    def _update_stop_orders(self):
        for pos in self.positions:
            if pos.handler and pos.pending_action:
                if pos.pending_action.position_change is None:
                    pos.handler.execute() # type: ignore
                    pos.handler, pos.pending_action = None, None
            
    # -------------------------------------------------------------------------    
    def __get_asset_precision(self, asset:str) -> int:
        all_symbols = self.broker.get_all_symbols(asset)
        
        if asset in self.broker.valid_quote_assets:
            symbol = next(
                filter(lambda x: x.quote_asset == asset, all_symbols),
                None
                )
            if symbol:
                return symbol.quote_precision
        
        else:
            symbol = next(
                filter(lambda x: x.base_asset == asset, all_symbols),
                None
                )
            if symbol:
                return symbol.base_precision
            
        raise ValueError(f'{asset} is not a valid asset')
             
