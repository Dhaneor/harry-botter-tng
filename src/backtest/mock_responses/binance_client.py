#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 10:58:53 2022

@author: dhaneor
"""
from datetime import datetime
from time import time, sleep
from random import random

# ==============================================================================
class Response:
    def __init__(self):
        self.headers = {'x-mbx-used-weight-1m' : 0,
                        'x-mbx-used-weight' : 0
                        }
        
    def set_headers(self, weight_short, weight_long):
        self.headers = {'x-mbx-used-weight-1m' : weight_short,
                        'x-mbx-used-weight' : weight_long,
                        } 
        
    def reset_headers(self):
        self.headers = {'x-mbx-used-weight-1m' : 0,
                        'x-mbx-used-weight' : 0
                        }

# ==============================================================================
class Client:
    def __init__(self, api_key:str, api_secret:str):
        self.weight_used = 0
        self.response = Response()
        self.latency = 0.01
        self.last_reset = int(time())
        self.reset_after = 60
    
    # ..........................................................................    
    def get_system_status(self):
        sleep(self.latency)
        self._increase_used_wweight()
        return {'status' : 'normal', 'message' : None}
    
    def get_historical_klines(self, symbol, interval, start_str, end_str, limit):
        sleep(self.latency)
        self._increase_used_wweight()
        return [int(time())]
    
    # ..........................................................................
    def _increase_used_wweight(self):
        self._reset_weight()
        self.weight_used += 1
        self._update_headers(self.weight_used)
    
    def _reset_weight(self):
        if time() - self.last_reset > self.reset_after:
            self.weight_used = 0
            self._update_headers(0)
            self.last_reset = int(time())
        
    def _update_headers(self, weight:int):
        self.response.set_headers(weight, weight)
        
    def _sleep(self):
        _rnd = random() / (self.latency / 5)
        _rnd = _rnd / 2 if _rnd < 0.5 else _rnd / -2
        sleep(self.latency + _rnd)
    
    
    
# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    c = Client(api_key='abc', api_secret='123')
    
    for _ in range(1000):
        c.get_system_status()
        print(f'{c.weight_used=} {c.last_reset=}')
        
    
    
    