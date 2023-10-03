#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 01 07:09:20 2022

@author dhaneor
"""
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import logging
from typing import Dict, Iterable, List, Tuple, Union

from src.broker.ganesh import Ganesh
from ..position_handlers.position_handler import PositionHandlerBuilder
from ..models.position import Position, PositionFactory
from ..models.requests import RequestFactory
from src.exchange.util.kucoin_async_client import AsyncKucoinClient
from data_sources.util.ws_kucoin import KucoinWebsocketPrivate, KucoinWebsocketPublic
from data_sources.util.publishers import CallbackPublisher

# =============================================================================
class Account:
    
    broker: Ganesh
    ws_client: KucoinWebsocketPrivate
    handler_builder = PositionHandlerBuilder()

    positions: dict[str, Position]
    debt_ratio: float
    
    def __init__(self, broker: Ganesh, quote_asset: str):
        self.broker = broker
        self.quote_asset: str = quote_asset 
               
        self.api_client = AsyncKucoinClient(credentials=broker.credentials)
        
        self.private_events_stream = KucoinWebsocketPrivate(
            credentials=self.broker.credentials,
            publisher=CallbackPublisher(callback=self.handle_event)
        )    
        self.public_events_stream = KucoinWebsocketPublic(
            publisher=CallbackPublisher(callback=self.handle_event)
        )
        self.position_factory = PositionFactory(self.broker) 
        self.request_factory = RequestFactory(quote_asset=quote_asset)
        
        self.logger = logging.getLogger('main.Account')

    def __repr__(self):
        
        return f'[Account] value: {self.get_account_value()} \t'\
            f'account leverage: {round(self.account_leverage, 2)} \t'\
                f'debt ratio: {round(self.debt_ratio, 2)}'
     
    # -------------------------------------------------------------------------
    async def run(self):
        await self.initialize_from_exchange_api()
        await self.private_events_stream.watch_account()
        
        interesting = [
            f'{asset}-{self.quote_asset}' \
                for asset, pos in self.positions.items() \
                    if pos.non_zero and pos.balance.asset != self.quote_asset
            ]   
        
        await self.public_events_stream.watch_candles(interesting, '1min')


        
    async def stop(self):        
        interesting = [
            f'{asset}-{self.quote_asset}' \
                for asset, pos in self.positions.items() \
                    if not pos.is_dust and pos.balance.asset != self.quote_asset
            ]        

        await self.public_events_stream.unwatch_candles(interesting, '1min')
        await self.ws_client.unwatch_account()
        
    async def initialize_from_exchange_api(self):
        positions = self.position_factory.build_positions(
            account = await self.api_client.get_margin_account(),
            quote_asset = self.quote_asset
        )
        
        self.positions = {item.asset: item for item in positions}

    async def handle_event(self, event: dict):
        if 'subject' in event:
            subject = event['subject']
            
            if subject == 'ticker':
                try:
                    asset = event['symbol'].split('-')[0]
                    price = event['data']['price']
                    self.positions[asset].last_price = price
                    log_str = f'{asset: <8} - {price}'
                    self.logger.debug(log_str)
                except Exception as e:
                    self.logger.exception(e)
                
                return
            
            elif subject == 'candle':
                try:
                    asset = event['symbol'].split('-')[0]
                    price = event['data']['close']
                    self.positions[asset].last_price = price
                except Exception as e:
                    self.logger.exception(e)
                
                return
            
            elif subject == 'account.balance':
                try:
                    asset = event['data']['currency']
                    position = self.positions[asset]
                    position.balance.update(event)
                except Exception as e:
                    self.logger.exception(e)
                    
                return
                    
            elif subject == 'debt.ratio':
                """
                example message:
                {
                    'type': 'message', 
                    'topic': '/margin/position', 
                    'userId': '5fd10f949910b40006395f9e', 
                    'channelType': 'private', 
                    'subject': 'debt.ratio', 
                    'data': {
                        'debtRatio': 0.6646934, 
                        'totalAsset': 0.000879960078878793, 
                        'totalDebt': '0.000584903655625', 
                        'debtList': {
                            'QNT': '0', 
                            'XRP': '0', 
                            'ETH': '0', 
                            'XLM': '0', 
                            'USDT': '10.0000625', 
                            'LINK': '0', 
                            'LTC': '0', 
                            'ADA': '0'
                    }, 
                    'timestamp': 1670178070087
                }
                """
                try:
                    self.debt_ratio = event['data']['debtRatio']
                    debt_list = event['data']['debtList']
                    if debt_list:
                        for asset, liability in debt_list.items():
                           try:
                               position = self.positions[asset]
                               position.balance.update_liability(liability)
                           except Exception as e:
                               self.logger.error(
                                   f'failed to update liabilty for {asset}: {e}'
                                )
                except Exception as e:
                    self.logger.exception(e)
                    
                return
         
            elif subject == 'stopOrder':
                """ 
                example message:
                {
                    'type': 'message', 
                    'topic': '/spotMarket/advancedOrders', 
                    'userId': '5fd10f949910b40006395f9e', 
                    'channelType': 'private', 
                    'subject': 'stopOrder', 
                    'data': {
                        'createdAt': 1670183280902, 
                        'orderId': 'vs93qoscv5o3qkrg000hhb61', 
                        'orderType': 'stop', 
                        'side': 'sell', 
                        'size': '9.8414', 
                        'stop': 'loss', 
                        'stopPrice': '0.39132', 
                        'symbol': 'XRP-USDT', 
                        'tradeType': 'MARGIN_TRADE', 
                        'ts': 1670183340836885829, 
                        'type': 'cancel'
                    }
                }
                """
                try:
                    symbol = event['data']['symbol']
                    asset = symbol.split('-')[0]
                    position = self.positions[asset]
                    position.handle_stop_order_update(event['data'])
                except Exception as e:
                    self.logger.exception(e)
                    
                return
            
            elif subject == 'orderChange':
                return
                                   
        self.logger.error(event)
            
        
    