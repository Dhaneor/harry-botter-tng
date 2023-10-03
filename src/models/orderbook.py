#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:57:18 2021

@author: dhaneor
"""

import time
import sys

from pprint import pprint


# ============================================================================
class Orderbook:

    def __init__(self, *args, **kwargs):

        self.symbol = kwargs.get('symbol') if kwargs.get('symbol') is not None else None
        self.bids = kwargs.get('bids') if kwargs.get('bids') is not None else None
        self.asks = kwargs.get('asks') if kwargs.get('asks') is not None else None

        # print(self)


    def __repr__(self):

        out = '-=*=-' * 30 + '\n'
        out += f'ORDERBOOK {self.symbol}:\n'
        if self.bids is not None:
            out += f'bids ({len(self.bids)}) {self.bids[:5]}\n'
        else: 
            out += f'bids (0) ´None´\n'
        if self.asks is not None:
            out += f'asks ({len(self.bids)}) {self.asks[:5]}\n'
        else: 
            out += f'asks (0) ´None´\n'
        out += '-=*=-' * 30 + '\n'

        return out
        
    # ------------------------------------------------------------------------- 
    def _get_orderbook(self):

        from exchange.binance_classic import Binance

        orderbook = None
        retry = 0
        max_retry = 2
        
        # fetch the currrent orderbook from Binance:
        with Binance() as conn:

            while orderbook is None:
                
                orderbook = conn.get_market_depth(symbol=self.symbol,
                                                  limit=500
                                                  )

                if retry > 0:  
                    print(f'Unable to get orderbook ({retry}): retrying ...')
                    time.sleep(1)

                if retry == max_retry: return False
                
                retry += 1

        # convet bids/asks to float and save in instance variable
        self.bids = self._convert_to_float(orderbook.get('bids'))
        self.asks = self._convert_to_float(orderbook.get('asks'))

        return 

    # take one side of the orderbook (bids/asks) and convert the values for
    # price and quantity from string (as delivered by Binance) to float        
    def _convert_to_float(self, side):

        for idx, elem in enumerate(side):

            # print(f'converting to float {elem[0]=} :: {elem[1]=}')
            side[idx] = [float(elem[0]), float(elem[1])]
            
        return side




# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #

if __name__ == '__main__':

    st = time.time()

    ob = Orderbook(symbol='ADABTC')
    # print(ob)

    



    # -------------------------------------------------------------------------

    print(round(time.time()-st, 5), ' seconds')