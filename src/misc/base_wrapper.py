#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 09 00:44:23 2025

@author dhaneor
"""

import numpy as np
from typing import Sequence, Union

from .numba_funcs import ffill_na_numba, cumsum_na_numba
from .mixins import PlottingMixin  # noqa: F401
from misc.exceptions import DimensionMismatchError


class BaseWrapper:
    """Common base class for BaseWrapper2D and BaseWrapper3D."""

    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if data.ndim not in [2, 3]:
            raise ValueError(f"Input data must be 2D or 3D. Received: {data.ndim}D.")

        self._data = data
        self.columns = list(columns)

    def __call__(self) -> np.ndarray:
        return self._data

    def __len__(self):
        return self._data.shape[0]

    def __iter__(self):
        return iter(self._data)

    def __sizeof__(self):
        return object.__sizeof__(self) + self._data.nbytes

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                col_index = self.columns.index(item)
            except ValueError:
                raise KeyError(f"Column '{item}' not found in {self.columns}.")
            return self._data[:, col_index]
        elif isinstance(item, (slice, int, np.integer)):
            return self._data[item]
        elif isinstance(item, tuple):
            return self._data[item]
        else:
            raise TypeError(f"Invalid index type ({type(item)}) for {item}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                col_index = self.columns.index(key)
            except ValueError:
                raise KeyError(f"Column '{key}' not found in {self.columns}.")
            self._data[..., col_index] = value

        elif isinstance(key, (slice, int, np.integer)):
            self._data[key] = value

        elif isinstance(key, tuple):
            self._data[key] = value

        else:
            raise TypeError(f"Invalid index type ({type(key)}) for {key}")

    def _binary_op(self, other, op):
        if isinstance(other, (int, float)):
            return self.__class__(op(self._data, other), self.columns)
        elif isinstance(other, self.__class__):
            return self.__class__(op(self._data, other._data), self.columns)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for operation")

    def __and__(self, other):
        return self._binary_op(other, np.logical_and)

    def __or__(self, other):
        return self._binary_op(other, np.logical_or)

    def __xor__(self, other):
        return self._binary_op(other, np.logical_xor)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def replace(
        self, old: Union[np.float64, np.int64], new: Union[np.float64, np.int64]
    ) -> None:
        self._data[self._data == old] = new

    def mean(self, axis=0):
        return np.nanmean(self._data, axis=axis)

    def std(self, axis=0):
        return np.nanstd(self._data, axis=axis)


class BaseWrapper2D(BaseWrapper):
    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        super().__init__(data, columns)
        if self._data.ndim != 2:
            raise ValueError("Input data must be 2D for BaseWrapper2D.")

    def ffill(self):
        self._data = ffill_na_numba(self._data)
        return self

    def cumsum(self):
        self._data = cumsum_na_numba(self._data)
        return self


class BaseWrapper3D(BaseWrapper):
    def __init__(self, data: np.ndarray, columns: Sequence[str], layers: Sequence[str]):
        super().__init__(data, columns)
        if self._data.ndim != 3:
            raise ValueError("Input data must be 3D for BaseWrapper3D.")
        self.layers = list(layers)
        if len(self.layers) != self._data.shape[2]:
            raise ValueError(
                f"Number of layers ({len(self.layers)}) must match the third dimension of data ({self._data.shape[2]})."
            )

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                col_index = self.columns.index(item)
            except ValueError:
                try:
                    layer_index = self.layers.index(item)
                    return self._data[:, :, layer_index]
                except ValueError:
                    raise KeyError(f"'{item}' not found in columns or layers.")
            return self._data[:, col_index, :]
        elif isinstance(item, (slice, int, np.integer)):
            return self._data[item]
        elif isinstance(item, tuple):
            return self._data[item]
        else:
            raise TypeError(f"Invalid index type ({type(item)}) for {item}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                col_index = self.columns.index(key)
                if value.ndim == 2 and value.shape == (
                    self._data.shape[0],
                    self._data.shape[2],
                ):
                    self._data[:, col_index, :] = value
                else:
                    raise DimensionMismatchError(
                        f"Invalid value shape for column assignment. Expected shape "
                        f"{(self._data.shape[0], self._data.shape[2])}, got {value.shape}."
                    )
            except ValueError:
                try:
                    layer_index = self.layers.index(key)
                    if value.ndim == 2 and value.shape == (
                        self._data.shape[0],
                        self._data.shape[1],
                    ):
                        self._data[:, :, layer_index] = value
                    else:
                        raise DimensionMismatchError(
                            f"Invalid value shape for layer assignment. Expected shape "
                            f"{(self._data.shape[0], self._data.shape[1])}, got {value.shape}."
                        )
                except ValueError:
                    raise KeyError(f"'{key}' not found in columns or layers.")
        elif isinstance(key, (slice, int, np.integer)):
            self._data[key] = value
        elif isinstance(key, tuple):
            self._data[key] = value
        else:
            raise TypeError(f"Invalid index type ({type(key)}) for {key}")

    def _binary_op(self, other, op):
        if isinstance(other, (int, float)):
            return self.__class__(op(self._data, other), self.columns, self.layers)
        elif isinstance(other, self.__class__):
            return self.__class__(op(self._data, other._data), self.columns, self.layers)
        else:
            raise TypeError(f"Invalid operand type ({type(other)}) for operation")


class SignalsWrapper(BaseWrapper3D):
    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        super().__init__(data, columns)
        self.symbols = self.columns
