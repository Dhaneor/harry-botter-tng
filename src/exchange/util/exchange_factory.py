#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 15:28:53 2022

@author: dhaneor
"""

from src.exchange.kucoin_ import KucoinFactory

# =============================================================================    
def get_exchange(exchange: str, market: str, credentials:dict):
    
    if exchange.lower() == 'kucoin':
        factory = KucoinFactory()
    else:
        raise ValueError(f'unknown exchange: {exchange}')
    
    return factory.build_client(market=market, credentials=credentials) 