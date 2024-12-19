#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:59:31 2021

@author: dhaneor
"""


# --------------------------------------------------------------------------------
# helper functions for import in different classes
def get_exchange_name(symbol:str) ->str:
        if '-' in symbol:
            return 'kucoin'
        else:
            return 'binance'





