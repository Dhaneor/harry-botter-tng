#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides some helper functions for handling Numpy arrays

Created on Sun Aug 06 01:28:53 2023

@author: dhaneor
"""
import numpy as np
import numpy.typing as npt
import numba as nb


# @nb.jit(nopython=True, cache=True)
def pct_change(arr: np.ndarray, axis=0, n: int = 1):
    """Calculates the percentage change between n-spaced values in an array.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    axis : int, optional
        Axis along which to calculate the percentage change. The default is 0.
    n : int, optional
        Percent change compared to n rows/columns ago. The default is 1.

    Returns
    -------
    pct_change : np.ndarray
        Array of percentage changes
    """
    out = np.empty_like(arr, dtype=np.float64)

    if axis == 0:
        out[n:] = (arr[n:] - arr[:-n]) / arr[:-n]
        out[:n] = np.nan
    elif axis == 1:
        out[:, n:] = (arr[:, n:] - arr[:, :-n]) / arr[:, :-n]
        out[:, :n] = np.nan
    else:
        raise ValueError('Axis must be 0 or 1')

    return out


@nb.jit(nopython=True, cache=True)
def pct_change(arr: np.ndarray, axis: int = 0, n: int = 1) -> npt.NDArray[np.float64]:
    """Calculates the percentage change between n-spaced values in an array.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    axis : int, optional
        Axis along which to calculate the percentage change. The default is 0.
    n : int, optional
        Percent change compared to n rows/columns ago. The default is 1.

    Returns
    -------
    pct_change : np.ndarray
        Array of percentage changes

    Raises
    ------
    ValueError
        If axis not in [0, 1]
    """
    out = np.empty_like(arr, dtype=np.float64)

    if axis == 0:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if i >= n:
                    out[i, j] = (arr[i, j] - arr[i-n, j]) / arr[i-n, j]
                else:
                    out[i, j] = np.nan
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if i >= n:
                    out[i, j] = (arr[i, j] - arr[i, j-n]) / arr[i, j-n]
                else:
                    out[i, j] = np.nan

    else:
        raise ValueError('Axis must be 0 or 1')

    return out
