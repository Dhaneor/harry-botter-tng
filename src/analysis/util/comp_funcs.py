#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the interface/abstract base class for all strategy classes,
and also a generic single/simple strategy class as well as a generic 
composite strategy class.

These are the building blocks for all concrete strategy implementations,
which are requested from and built by the strategy builder, based on
standardized strategy definitions.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""
import numpy as np
from numpy import typing as npt
from numba import njit


def is_above(
    arr: np.ndarray, 
    x: np.float_ | np.int_
) -> npt.NDArray[np.bool_]:
    """Check whether values in array are above a certain value.

    Parameters
    ----------
    arr : np.ndarray
        one- or two-dimensional array of values
    x : np.float_
        value to check against

    Returns
    -------
    np.ndarray[np.bool_]
        resulting array of boolean values
    """
    return (arr > x)

@njit(nogil=True) 
def is_above_nb(arr: np.ndarray, x: np.ndarray | float | int) -> bool:
    out = np.full_like(arr, np.nan, dtype=np.bool_)
    
    for i in range(len(arr)):
        if isinstance(x, (float, int)):
            out[i] = True if arr[i] > x else False
        else:
            out[i] = True if arr[i] > x[i] else False
    return out

def is_above_or_equal():
    raise NotImplementedError('someone´s been lazy here ...')
    
def is_below(
    arr: np.ndarray, 
    x: np.float_ | np.int_
) -> npt.NDArray[np.bool_]:
    """Check whether values in array are below a certain value.

    Parameters
    ----------
    arr : np.ndarray
        one- or two-dimensional array of values
    x : np.float_
        value to check against

    Returns
    -------
    np.ndarray[np.bool_]
        resulting array of boolean values
    """
    return (arr < x)

def is_below_or_equal():
    raise NotImplementedError('someone´s been lazy here ...')

def is_equal(
    arr: np.ndarray, 
    x: npt.ArrayLike | np.float_ | np.int_ | np.bool_
) -> np.ndarray[np.bool_]:
    """Check whether values in array are below a certain value.

    Parameters
    ----------
    arr : np.ndarray
        one- or two-dimensional array of values
    x : np.float_
        value to check against

    Returns
    -------
    np.ndarray[np.bool_]
        resulting array of boolean values
    """
    return (arr == x)

def is_not_equal():
    raise NotImplementedError('someone´s been lazy here ...')
         
def crossed_above(
    arr1: np.ndarray, 
    arr2: np.ndarray | float | int
) -> np.ndarray:
    """Checks whether one time series values crossed above value.

    arr1 represents the first time series values, arr2 represents 
    the second time series values or a fixed value. The rows of the
    array(s) represent one observation, while the columns contain
    all observations for one asset, or another time series in 
    general.
    
    For each cell in the output array, the resultiung value is True 
    if the first time series value is larger than the value for the 
    same index in time series in arr2 (or the scalar value), and the 
    value in the preceding row is not. 

    Parameters
    ----------
    arr1 : np.ndarray
        time-series values as columns of array 
    arr2 : np.ndarray | float | int
        time series values as columns of array _or_ a scalar value

    Returns
    -------
    np.ndarray
        array with: True if values crossed above, False otherwise

    Raises
    ------
    ValueError
        if arr2 is neither an int/float nor an array
    """
    result = np.empty_like(arr1, dtype=bool)

    if isinstance(arr2, (float, int)):
        result[1:] = np.logical_and(
            arr1[:-1] <= arr2, 
            arr1[1:] > arr2
        )
    elif isinstance(arr2, np.ndarray):
        result[1:] = np.logical_and(
            arr1[:-1] <= arr2[:-1], 
            arr1[1:] > arr2[1:]
        )
    else:
        raise ValueError("arr2 must be either a float/int or an array")

    return result

def crossed_below(
    arr1: np.ndarray, 
    arr2: np.ndarray | float | int
) -> np.ndarray:
    """Checks whether one time series values crossed above value.

    arr1 represents the first time series values, arr2 represents 
    the second time series values or a fixed value. The rows of the
    array(s) represent one observation, while the columns contain
    all observations for one asset, or another time series in 
    general.
    
    For each cell in the output array, the resultiung value is True 
    if the first time series value is larger than the value for the 
    same index in time series in arr2 (or the scalar value), and the 
    value in the preceding row is not. 

    Parameters
    ----------
    arr1 : np.ndarray
        time-series values as columns of array 
    arr2 : np.ndarray | float | int
        time series values as columns of array _or_ a scalar value

    Returns
    -------
    np.ndarray
        array with: True if values crossed above, False otherwise

    Raises
    ------
    ValueError
        if arr2 is neither an int/float nor an array
    """
    result = np.zeros_like(arr1, dtype=bool)
    
    if isinstance(arr2, (float, int)):
        result[1:] = np.logical_and(
            arr1[:-1] >= arr2, 
            arr1[1:] < arr2
        )
    elif isinstance(arr2, np.ndarray):
        result[1:] = np.logical_and(
            arr1[:-1] >= arr2[:-1], 
            arr1[1:] < arr2[1:]
            )
    else:
        raise ValueError("arr2 must be either a float or an array")

    return result

def crossed():
    raise NotImplementedError('someone´s been lazy here ...')
   