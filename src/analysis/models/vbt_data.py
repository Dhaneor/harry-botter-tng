#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 16:49:20 2022

@author dhaneor
"""
from typing import Union, Iterable
from pprint import pprint
from functools import partial
import concurrent.futures
import datetime as dt
import dateparser as dp
from vectorbtpro import Data, CCXTData
from vectorbtpro.utils.datetime_ import to_tzaware_datetime

from util.timeops import interval_to_milliseconds

LOOKBACK = 500

# =============================================================================
class LiveData(CCXTData):

    lookback: int = 300
    max_workers: int  = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def fetch_symbol_multiple(cls,
                              exchange: str,
                              symbols: Union[str, Iterable[str]],
                              interval: str):
        exchange = cls.resolve_exchange(exchange)

        # find start and end date for period of the most recent 1000
        # values for the given timeframe/interval
        end = dt.datetime.now()
        start = end - ((end - dp.parse(interval)) * cls.lookback) #type:ignore
        start = to_tzaware_datetime(start, tz='UTC')
        end = to_tzaware_datetime(end, tz='UTC')

        # partial for use with 'map' in executor below
        map_func = partial(
            cls.fetch_symbol,
            exchange=exchange, timeframe=interval, start=start, end=end,
            limit=cls.lookback, show_progress=True, silence_warnings=True
        )

        # download the data in parallel with threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cls.max_workers
        ) as executor:
            results = executor.map(map_func, symbols)

        return {sym: res for sym, res in zip(symbols, results)}

    def update(self, update: dict):
        pass