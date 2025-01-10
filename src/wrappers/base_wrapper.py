#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 09 00:44:23 2025

@author dhaneor
"""
import numpy as np
import bottleneck as bn
from collections.abc import Iterable
from typing import Sequence, Union

from .numba_funcs import (
    apply_to_columns, ffill_na_numba, cumsum_na_numba
    )


# .................................. Numba functions ..................................
# @njit
# def ffill_na_numba(arr):
#     if arr.ndim == 2:
#         for col in range(arr.shape[1]):
#             last_valid = arr[0, col]
#             for row in range(arr.shape[0]):
#                 if np.isnan(arr[row, col]):
#                     arr[row, col] = last_valid
#                 else:
#                     last_valid = arr[row, col]
#     elif arr.ndim == 3:
#         for i in range(arr.shape[0]):
#             for col in range(arr.shape[2]):
#                 last_valid = arr[i, 0, col]
#                 for row in range(arr.shape[1]):
#                     if np.isnan(arr[i, row, col]):
#                         arr[i, row, col] = last_valid
#                     else:
#                         last_valid = arr[i, row, col]
#     else:
#         raise ValueError("Input array must be 2D or 3D")
#     return arr


# ................................. BaseWrapper Class .................................
class BaseWrapper:

    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        # Check if data is a numpy array
        if not isinstance(data, np.ndarray):
            # Check if data is an Iterable but not a string
            if isinstance(data, Iterable) and not isinstance(data, str):
                try:
                    data = np.array(data)
                    # If the data is 1D, reshape it to 2D
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                except ValueError:
                    raise ValueError(
                        "Input data must be convertible to a numpy array. "
                        f"Received: {type(data)}."
                    )
            else:
                raise ValueError(
                    "Input data must be a numpy array or an Iterable (not a string). "
                    f"Received: {type(data)}."
                )
        # check for dimensions, the wrapper should only accept 2D or 3D numpy arrays
        if data.ndim not in [2, 3]:
            raise ValueError("Input data must be 2D or 3D. Received: {data.ndim}D. ")

        self.data = data
        self.columns: list[str] = columns

    def __call__(self) -> object:
        return self.data

    def __len__(self):
        return self.data.shape[0]
    
    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        if isinstance(item, str):
            for idx, col in enumerate(self.columns):
                if item == col:
                    return self.data[:, idx]
        elif isinstance(item, (slice, int)):
            return self.data[item]
        elif isinstance(item, tuple):
            return self.data[item]
        else:
            raise TypeError(f"Invalid index type ({type(item)}) for {item}")
        
    def __setitem__(self, key, value):
        if isinstance(key, str):
            for idx, col in enumerate(self.columns):
                if key == col:
                    self.data[:, idx] = value
                    return
            raise KeyError(f"Column '{key}' not found")
        elif isinstance(key, (slice, int)):
            self.data[key] = value
        elif isinstance(key, tuple):
            self.data[key] = value
        else:
            raise TypeError(f"Invalid index type ({type(key)}) for {key}")

    def __and__(self, other: Union["BaseWrapper", int, float]) -> "BaseWrapper":
        if isinstance(other, BaseWrapper):
            if self.columns != other.columns:
                raise ValueError("Symbols must match for intersection")
            return BaseWrapper(np.logical_and(self(), other()), self.columns)
        elif isinstance(other, (int, float, bool)):
            return BaseWrapper(np.logical_and(self(), other), self.columns)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for intersection")

    def __or__(self, other: Union["BaseWrapper", int, float]) -> "BaseWrapper":
        if isinstance(other, BaseWrapper):
            if self.columns != other.columns:
                raise ValueError("Symbols must match for union")
        
            return BaseWrapper(np.logical_or(self(), other()), self.columns)
        elif isinstance(other, (int, float, bool)):
            return BaseWrapper(np.logical_or(self(), other), self.columns)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for union")

    def __xor__(self, other: Union["BaseWrapper", int, float]) -> "BaseWrapper":
        if isinstance(other, BaseWrapper):
            if self.columns != other.columns:
                raise ValueError("Symbols must match for XOR")
            return BaseWrapper(np.logical_xor(self(), other()), self.columns)
        elif isinstance(other, (int, float, bool)):
            return BaseWrapper(np.logical_xor(self(), other), self.columns)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for XOR")

    def __add__(self, other: Union["BaseWrapper", int, float]) -> "BaseWrapper":
        if isinstance(other, BaseWrapper):
            if self.columns != other.columns:
                raise ValueError("Symbols must match for addition")
            return BaseWrapper(np.add(self(), other()), self.columns)
        elif isinstance(other, (int, float)):
            return BaseWrapper(np.add(self(), other), self.columns)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for addition")

    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim

    def replace(self, old: np.float_ | np.int_, new: np.float_ | np.int_) -> None:
        try:
            bn.replace(self.data, old, new)
        except TypeError as e:
            print(f"Error replacing values: {e}")
            print(type(self.data))

    def ffill(self):
        self.data = ffill_na_numba(self.data)
        return self

    def cumsum(self):
        self.data = cumsum_na_numba(self.data)
        return self
    
    def mean(self, axis=0) -> float:
        return bn.nanmean(self.data, axis=0)
    
    def std(self) -> float:
        return bn.nanstd(self.data, axis=0)


class SignalsWrapper(BaseWrapper):

    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        super().__init__(data, columns)
        self.symbols = self.columns