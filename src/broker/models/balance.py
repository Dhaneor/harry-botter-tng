#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:23:58 2022

@author: dhaneor
"""
import logging
from dataclasses import dataclass
from typing import Union

logger = logging.getLogger('main.balance')

# ==============================================================================
@dataclass
class Balance:
    asset: str
    free: float
    locked: float
    borrowed: float
    total: float
    
    def __repr__(self) -> str:
        out = f'[Balance] {self.asset: <8}: {self.net: 14} \ttotal:'\
        f' {self.total: 14} (free:{self.free: 14} locked:{self.locked: 14})'\
        f'\tborrowed:{self.borrowed: <14}'
        
        return out
    
    def __post_init__(self):
        self.logger = logging.getLogger(f'main.Balance.{self.asset}')
        self.logger.setLevel(logging.DEBUG)
    
    @property
    def net(self):
        return self.total - self.borrowed
    
    @property
    def is_zero(self):
        return self.net == 0
    
    def update(self, msg: dict):
        """Updates the balance based on the message contents.
        
        :param msg:  {
            'id': '638d2f944bbdac0001040d0f', 
            'type': 'message', 
            'topic': '/account/balance', 
            'userId': '5fd10f949910b40006395f9e', 
            'channelType': 'private', 
            'subject': 'account.balance', 
            'data': {
                'accountId': '5ffa704f1839110006130f1e', 
                'available': '1.08368592345990867', 
                'availableChange': '0.358323573744', 
                'currency': 'USDT', 
                'hold': '0', 
                'holdChange': '0', 
                'relationContext': {
                    'symbol': 'ADA-USDT', 
                    'orderId': '638d2f94a264690001115187', 
                    'tradeId': '1391073992261633'
                }, 
                'relationEvent': 'margin.setted', 
                'relationEventId': '638d2f944bbdac0001040d0f', 
                'time': '1670197140846', 
                'total': '1.08368592345990867'
            }
        }

        :type message: dict
        :return: None
        :rtype: None
        """
        try:
            data = msg['data']
            self.free = float(data['available'])
            self.locked = float(data['hold'])
            self.total = float(data['total'])
            event = data['relationEvent'] if 'relationEvent' in data else 'unknown'
            time = data['time']
            rc = data['relationContext']
            self.logger.info(f'updated balance for {event} event at {time}: {self}\n\t\t\t\t\t{rc}')
        except Exception as e:
            self.logger.error(
                f'balance update failed - message: {msg}, error: {e}'
            )
            
    def update_liability(self, value: Union[float, str]):
        if float(value) !=  self.borrowed:
            self.borrowed = float(value)
            self.logger.debug(f'updated liability: {self}')


# ==============================================================================
def balance_factory(balance: dict) -> Balance:
    if 'asset' in balance:
        return Balance(
            asset=balance.get('asset', ''),
            free=balance.get('free', 0),
            locked=balance.get('locked', 0),
            borrowed=balance.get('borrowed', 0),
            total=balance.get('total', 0)
        )
    else:
        return Balance(
            asset=balance.get('currency', ''),
            free=float(balance.get('availableBalance', 0)),
            locked=float(balance.get('holdBalance', 0)),
            borrowed=float(balance.get('liability', 0)),
            total=float(balance.get('totalBalance', 0))
        )
