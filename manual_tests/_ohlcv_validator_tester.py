#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23 10:55:23 2022

@author dhaneor
"""

import sys
import os
import time
import logging

from pprint import pprint
from typing import Iterable, Union

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from staff.hermes import Hermes
from analysis.util.ohlcv_validator import OhlcvValidator
from util.timeops import execution_time


LOGGER = logging.getLogger('main')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

HERMES = Hermes(exchange='kucoin', verbose=True)
OV = OhlcvValidator()

# =============================================================================
def get_ohlcv(symbol:str, interval:str, start:Union[int, str],
              end:Union[int, str]) -> dict:
    return HERMES.get_ohlcv(
        symbols=symbol, interval=interval, start=start, end=end
    )


# -----------------------------------------------------------------------------
@execution_time
def test_find_missing_rows_in_df(symbol:str, interval:str, start:Union[int, str],
                                 end:Union[int, str]):

    query_result = get_ohlcv(
        symbol=symbol, interval=interval, start=start, end=end
    )

    if query_result.get('success'):
        df = query_result.get('message')
        print(df)

        missing = OV.find_missing_rows_in_df(
            df=df, interval=interval, start=start, end=end
        )

        pprint(missing)

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    test_find_missing_rows_in_df(
        symbol='BTC-USDT', interval='15m', start=-20000,
        end='March 12, 2021 05:15:00'
    )