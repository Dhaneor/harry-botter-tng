#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:53:23 2025

@author_ dhaneor
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def apply_to_columns(arr: np.ndarray, func):
    if arr.ndim == 2:
        for col in range(arr.shape[1]):
            arr[:, col] = func(arr[:, col])
    elif arr.ndim == 3:
        for layer in prange(arr.shape[2]):
            for col in range(arr.shape[1]):
                arr[:, col, layer] = func(arr[:, col, layer])
    else:
        raise ValueError("Input array must be 2D or 3D")
    return arr


@njit(parallel=True)
def apply_to_columns_general(arr: np.ndarray, func):
    # Determine the number of dimensions
    ndim = arr.ndim
    if ndim < 1:
        raise ValueError("Input array must have at least 1 dimension")
    
    # Calculate the total number of columns across all other dimensions
    num_cols = 1
    for dim in range(1, ndim):
        num_cols *= arr.shape[dim]
    
    # Reshape the array to 2D: (axis=0, all other axes flattened)
    shape_0 = arr.shape[0]
    shape_other = num_cols
    arr_reshaped = arr.reshape(shape_0, shape_other)
    
    # Apply the function to each column in parallel
    for col in prange(shape_other):
        column = arr_reshaped[:, col]
        arr_reshaped[:, col] = func(column)
    
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
