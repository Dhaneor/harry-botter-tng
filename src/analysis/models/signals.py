#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a signals JIT class to store and transform signals.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""

import numpy as np
import numpy.typing as npt
from numba import njit, float32
from numba.experimental import jitclass
from typing import Optional


@njit
def combine_signals(
    open_long: np.ndarray,
    close_long: np.ndarray,
    open_short: np.ndarray,
    close_short: np.ndarray,
):
    n_periods, n_assets, n_strats = open_long.shape
    positions_out = np.zeros((n_periods, n_assets, n_strats), dtype=np.float32)

    for t in range(n_periods):
        for a in range(n_assets):
            for s in range(n_strats):
                if t == 0:
                    if open_long[t, a, s] == 1:
                        positions_out[t, a, s] = 1
                    elif open_short[t, a, s] == 1:
                        positions_out[t, a, s] = -1
                else:
                    positions_out[t, a, s] = positions_out[t - 1, a, s]

                    if positions_out[t, a, s] == 1:
                        if close_long[t, a, s] == 1 or open_short[t, a, s] == 1:
                            positions_out[t, a, s] = 0
                    elif positions_out[t, a, s] == -1:
                        if close_short[t, a, s] == 1 or open_long[t, a, s] == 1:
                            positions_out[t, a, s] = 0

                    if positions_out[t, a, s] == 0:
                        if open_long[t, a, s] == 1:
                            positions_out[t, a, s] = 1
                        elif open_short[t, a, s] == 1:
                            positions_out[t, a, s] = -1

    return positions_out


spec = [
    ("data", float32[:, :, :]),
]


@jitclass(spec)
class SignalStore:

    def __init__(self, data: npt.ArrayLike):
        self.data = data

    def __add__(self, other):
        if isinstance(other, SignalStore):
            return SignalStore(np.add(self.data, other.data))
        elif isinstance(other, (float, int)):
            return SignalStore(np.add(self.data, np.float32(other)))
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

    def __radd__(self, other):
        return self.__add__(other)


class Signals:
    def __init__(
        self,
        symbols: list[str],
        open_long: np.ndarray,
        close_long: np.ndarray,
        open_short: np.ndarray,
        close_short: np.ndarray,
    ):
        self.symbols = symbols
        self._store = SignalStore(open_long, close_long, open_short, close_short)

    def __add__(self, other) -> 'Signals':
        if isinstance(other, Signals):
            if self.symbols != other.symbols:
                raise ValueError("Symbols must match for addition")
            new_store = self._store + other._store
        elif isinstance(other, (float, int)):
            new_store = self._store + other
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

        return Signals(symbols=self.symbols, store=new_store)
    
    def __radd__(self, other) -> 'Signals':
        return self.__add__(other)
    
    def __len__(self) -> int:
        return self._store.data.shape[0]
    
    def __iter__(self):
        return iter(self._store.data)

    # ..................................................................................
    @property
    def data(self):
        return self._store.data

    # ..................................................................................
    @classmethod
    def from_separate(
        cls,
        symbols: list[str],
        open_long: Optional[np.ndarray] = None,
        close_long: Optional[np.ndarray] = None,
        open_short: Optional[np.ndarray] = None,
        close_short: Optional[np.ndarray] = None,
    ):
        shape = (
            (len(symbols), len(symbols[0]))
            if any([open_long, close_long, open_short, close_short])
            else (0, 0)
        )

        open_long = np.zeros(shape) if open_long is None else open_long
        close_long = np.zeros(shape) if close_long is None else close_long
        open_short = np.zeros(shape) if open_short is None else open_short
        close_short = np.zeros(shape) if close_short is None else close_short

        combined_signals = combine_signals(
            open_long, close_long, open_short, close_short
        )
        return cls(symbols, combined_signals)
