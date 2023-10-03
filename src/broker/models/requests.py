#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August 18  21:17:23 2022

@author: dhaneor
"""
import logging
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Iterable

logger = logging.getLogger('main.requests')
logger.setLevel(logging.INFO)

# ==============================================================================
@dataclass
class PositionChangeRequest:
    asset: str
    target: float = 0
    stop_loss: Union[tuple, None] = None
    take_profit: Union[tuple, None] = None
    quote_asset: Union[str, None]= None
    
# ==============================================================================
class RequestFactory:
    
    def __init__(self, quote_asset: str):
        self.quote_asset: str = quote_asset
        self._requests: Dict[str, PositionChangeRequest] = {}
        
        logger_name = 'main.requests.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
    
    # --------------------------------------------------------------------------
    def prepare_requests(self, target_account: List[dict], 
                         all_assets: Iterable[str]):

        for asset in all_assets:
            target_request = next(
                filter(lambda x: x.get('asset') == asset, target_account), None
                )
            
            self._requests[asset] = self._build_request_object(
                asset, target_request
                ) 
        
    def get_request_for_asset(self, asset) -> Union[PositionChangeRequest, None]:
        try:
            return self._requests[asset]
        except:
            return None
            
    # --------------------------------------------------------------------------                
    def _build_request_object(self, asset: str, target: Union[dict, None]
                              ) -> PositionChangeRequest:
        
        if target:
            
            self.logger.debug(target)
            
            request = PositionChangeRequest(
                asset=target['asset'],
                target=target['target'],
                stop_loss=target.get('stop_loss'),
                take_profit=target.get('take_profit'),
                quote_asset=target.get('quote_asset')
            )
            
            if request.stop_loss and request.target == 0:
                request.stop_loss = None 
                warning = f'position change request for {request.asset} had '
                warning += f'stop loss values, but target amount = 0'
                warning += f' - removed stop loss values'
                self.logger.warning(warning)
                
            if request.stop_loss and request.target == 0:
                request.take_profit = None 
                warning = f'position change request for {request.asset} had '
                warning += f'take profit values, but target amount = 0'
                warning += f' - removed take profit values'
                self.logger.warning(warning)
                
            return request

        else:
            return PositionChangeRequest(
                asset=asset, quote_asset=self.quote_asset
            )
    