#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 02 19:56:23 2022

@author: dhaneor
"""
import asyncio
from asyncio import create_task
from typing import Union, Optional, List, Tuple

from src.exchange.util.kucoin_async.client import Trade
from helpers.timeops import get_start_and_end_timestamp

# ==============================================================================
class KucoinOrderDownloader:
    """Downloads multiple orders from Kucoin.
    
    This class is used for the flexible handling of download requests 
    for multiple orders and handles the splitting into multiple time
    periods and the paging of requests if necessary.
    """
    
    def __init__(self, client: Trade):   
        self.client = client
        
    async def get_orders(
        self,
        start:Union[int, str], end:Union[int, str], trade_type: str,
        symbol: Optional[str]=None, side: Optional[str]=None, 
        order_type: Optional[str]=None, status: Optional[str]=None,
        ) -> List[dict]:
   
        periods, tasks, orders  = self._get_download_periods(start, end), [], []

        # Splitting the request is necessary if the period between
        # start and end date is longer than seven days. This is a
        # Kucoin API restriction. 
        for p in periods:    
            kwargs = {
                'symbol' : symbol,
                'side' : side.lower() if side else None,
                'type' : order_type.lower() if order_type else None,
                'status' : status,
                'tradeType' : trade_type,
                'startAt' : p[0],
                'endAt' : p[1]
            }
            tasks.append(
                create_task(self._get_orders_for_one_period(**kwargs))
            )
            
        orders = await asyncio.gather(*tasks)

        # we now have a list of lists and need to flatten it      
        return [item for sub_list in orders for item in sub_list]
    
    async def _get_orders_for_one_period(self, *args, **kwargs
                                         ) -> Union[List[dict], None]:
        # at some point in time Kucoin started to reject None as
        # parameter value, so we have to remove these here
        kwargs = {k: v for k,v in kwargs.items() if v is not None}
        kwargs['pageSize'] = 500
        orders, page = [], 1
        
        while True:
            kwargs['page'] = page
            
            res = await self.client.get_order_list(**kwargs)      
            orders += [item for item in res.get('items')]

            if page >= res.get('totalPage'):
                return orders
            page += 1

    def _get_download_periods(self, start: Union[str, int], 
                              end: Union[str, int]) -> tuple:
        
        start_ts, end_ts = self._get_start_and_end_timestamp(start, end) 
        timestamps, done = [], False
        seven_days = 3600 * 24 * 7 * 1000
        
        if end_ts - start_ts > seven_days: 
            while not done:
                _end = start_ts + seven_days - 1
                chunk = (start_ts, _end) if _end < end_ts else (start_ts, end_ts)
                timestamps.append(chunk)
                done = True if chunk[1] >= end_ts else False
                start_ts += seven_days  
        else:
            timestamps = [(start_ts, end_ts)]
            
        return tuple(timestamps)
        
    def _get_start_and_end_timestamp(self, start:Union[str, int], 
                                    end:Union[str, int]) -> tuple:
    
        # get the timestamps for start and/or end
        (start_ts, end_ts) = get_start_and_end_timestamp(start, end, '1d', 
                                                         unit='milliseconds', 
                                                         verbose=True)

        # if no start date was given, we download the last seven days,
        # which is the maximum period that Kucoin gives us in one go.
        if start is None:
            seven_days = 3600 * 24 * 7 * 1000 
            start_ts = end_ts - seven_days 
            
        return start_ts, end_ts
