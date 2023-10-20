#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import numpy as np
import pandas as pd
import talib as ta
import time
from pprint import pprint
from tqdm import trange

# profiler imports 
from cProfile import Profile
from pstats import SortKey, Stats

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import src.analysis.util.numpy_helpers as nph
import src.helpers.timeops as to


arr = np.random.rand(1000, 1000)
# initialize function = numba compilation
nph.pct_change_nb(arr, axis=0, n=1)
nph.pct_change_nb2(arr, axis=0, n=1)



def test_pct_change(runs=1):
    st = time.time()
    for i in range(runs):
        res = nph.pct_change(arr, axis=1, n=1)
    et = time.time()
    print(f'avg time: {1_000_000*(et-st)/runs:.2f}µs')
    return res

def test_pct_change_ta(runs=1):
    st = time.time()
    
    res = np.empty_like(arr)
    
    for _ in range(runs):
        for i in range(arr.shape[1]):
            res[:, i] = ta.ROCP(arr[:, i], 1)
    et = time.time()
    print(f'avg time: {1_000_000*(et-st)/runs:.2f}µs')
    return res



# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    res = test_pct_change(1)
    
    df = pd.DataFrame({'values': arr[:, 0], 'numpy': res[:, 0]})
    df['pandas'] = df['values'].pct_change()
    df['talib'] = ta.ROCP(arr[:, 0], 1)
    print(df.head(10))
    
    arr2 = np.random.rand(10,)
    print(arr2.shape)
    print(ta.ROC(arr2, 1))
    
    with Profile(timeunit=.000_001) as p:
        for _ in range(1_000):
            test_pct_change()
     
    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE) #(SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )