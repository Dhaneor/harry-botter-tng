#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:53:23 2025

@author_ dhaneor
"""

import numpy as np
from numba import njit


@njit
def apply_to_columns(arr: np.ndarray, func):
    if arr.ndim == 2:
        for col in range(arr.shape[1]):
            arr[:, col] = func(arr[:, col])
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            for col in range(arr.shape[2]):
                arr[i, :, col] = func(arr[i, :, col])
    else:
        raise ValueError("Input array must be 2D or 3D")
    return arr


@njit
def ffill_column(col: np.ndarray):
    last_valid = col[0]
    for i in range(len(col)):
        if np.isnan(col[i]):
            col[i] = last_valid
        else:
            last_valid = col[i]
    return col


@njit
def cumsum_column(col: np.ndarray):
    return np.cumsum(col)


@njit
def ffill_na_numba(arr: np.ndarray):
    return apply_to_columns(arr, ffill_column)


@njit
def cumsum_na_numba(arr):
    return apply_to_columns(arr, cumsum_column)
