#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 28 20:08:20 2023

@author dhaneor
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def atr(open_, high, low, period=14):
    """Caclulates ATR time series/array (Numba accelerated).

    :param open_: time series of open prices
    :type open_: np.nd_array
    :param high: time series of high prices
    :type high: np.nd_array
    :param low: time series of low prices
    :type low: np.nd_array
    :param period: ATR lookback period, defaults to 14
    :type period: int, optional
    :return: time series of ATR values
    :rtype: np.nd_array
    """
    atr = np.zeros(len(open_))
    for i in range(period, len(open_)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - open_[i]),
            abs(low[i] - open_[i])
        )
        atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr
