#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a signals JIT class to store and transform signals.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""

import itertools
import logging
import numpy as np
import numpy.typing as npt
from numba import njit, float32, from_dtype, types, typeof
from numba.experimental import jitclass

from analysis.dtypes import SIGNALS_DTYPE
from util.proj_types import SignalsArrayT
from misc.base_wrapper import BaseWrapper3D
from misc.numba_funcs import ffill_na_numba

logger = logging.getLogger("main.signals")

# define Numba type for signal records
SignalRecord = from_dtype(SIGNALS_DTYPE)
SignalArray3D = types.Array(SignalRecord, 3, "C")


def generate_test_data(periods=8, num_symbols=1, num_strategies=1) -> np.ndarray:
    base_patterns = {
        "open_long": (1, 0, 0, 0, 1, 0, 0, 0),
        "close_long": (0, 1, 0, 0, 0, 0, 0, 0),
        "open_short": (0, 0, 1, 0, 0, 0, 1, 0),
        "close_short": (0, 0, 0, 1, 0, 0, 0, 0),
        "combined": (1, 0, -1, 0, 1, 1, -1, -1),
    }

    def create_array(pattern):
        cycle = itertools.cycle(pattern)
        return (
            np.array([next(cycle) for _ in range(periods)])
            .reshape(periods, 1, 1)
            .astype(np.float32)
        )

    out = np.empty((periods, 1, 1), dtype=SIGNALS_DTYPE)

    for key in base_patterns.keys():
        out[key] = create_array(base_patterns[key])

    return np.tile(out, (1, num_symbols, num_strategies))


# ............................ Functions to combine signals ............................
@njit
def combine_signals(signals: typeof(SignalArray3D)) -> np.ndarray:  # type: ignore
    """Combines signals from 3D array into a single combined signal.

    Expects a 3D array with the custom dtype for signal records.

    The function returns a new 3D array with the combined signals,
    which saves memory and is the format required by the SignalStore
    class (see below).

    Parameters
    ----------
    signals : np.ndarray (SIGNALS_DTYPE)

    Returns
    -------
    np.ndarray (np.float32)
    """
    n_periods, n_assets, n_strats = signals.shape

    out = np.zeros((n_periods, n_assets, n_strats), dtype=np.float32)

    open_long = signals["open_long"]
    close_long = signals["close_long"]
    open_short = signals["open_short"]
    close_short = signals["close_short"]

    for t in range(n_periods):
        for a in range(n_assets):
            for s in range(n_strats):
                out[t, a, s] = 0
                if t == 0:
                    if open_long[t, a, s] == 1:
                        out[t, a, s] = 1
                    elif open_short[t, a, s] == 1:
                        out[t, a, s] = -1
                else:
                    out[t, a, s] = out[t - 1, a, s]

                    if out[t, a, s] == 1:
                        if close_long[t, a, s] == 1 or open_short[t, a, s] == 1:
                            out[t, a, s] = 0
                    elif out[t, a, s] == -1:
                        if close_short[t, a, s] == 1 or open_long[t, a, s] == 1:
                            out[t, a, s] = 0

                    if out[t, a, s] == 0:
                        if open_long[t, a, s] == 1:
                            out[t, a, s] = 1
                        elif open_short[t, a, s] == 1:
                            out[t, a, s] = -1

    return out


@njit
def combine_signals_3D(signals: typeof(SignalArray3D)) -> SignalArray3D:  # type: ignore
    """Combines signals from 3D array into a single combined signal.

    Expects a 3D array with the custom dtype for signal records.

    Different from the previous function, this version modifies the
    'combined' field of the elements of the original 3D array.

    Parameters
    ----------
    signals : np.ndarray (SIGNALS_DTYPE)

    Returns
    -------
    np.ndarray (SIGNALS_DTYPE)
    """
    n_periods, n_assets, n_strats = signals.shape

    open_long = signals["open_long"]
    close_long = signals["close_long"]
    open_short = signals["open_short"]
    close_short = signals["close_short"]

    for t in range(n_periods):
        for a in range(n_assets):
            for s in range(n_strats):
                signals["combined"][t, a, s] = 0
                if t == 0:
                    if open_long[t, a, s] == 1:
                        signals["combined"][t, a, s] = 1
                    elif open_short[t, a, s] == 1:
                        signals["combined"][t, a, s] = -1
                else:
                    signals["combined"][t, a, s] = signals["combined"][t - 1, a, s]

                    if signals["combined"][t, a, s] == 1:
                        if close_long[t, a, s] == 1 or open_short[t, a, s] == 1:
                            signals["combined"][t, a, s] = 0
                    elif signals["combined"][t, a, s] == -1:
                        if close_short[t, a, s] == 1 or open_long[t, a, s] == 1:
                            signals["combined"][t, a, s] = 0

                    if signals["combined"][t, a, s] == 0:
                        if open_long[t, a, s] == 1:
                            signals["combined"][t, a, s] = 1
                        elif open_short[t, a, s] == 1:
                            signals["combined"][t, a, s] = -1

    return signals


@njit
def combine_signals_np(signals: typeof(SignalArray3D)) -> np.ndarray:  # type: ignore
    """Combines signals from 3D array into a single combined signal.

    NOTE: This function is only for testing purposes and is much
    slower than the combine_signals function above

    Expects a 3D array with the custom dtype for signal records.

    The function returns a new 3D array with the combined signals,
    which saves memory and is the format required by the SignalStore
    class (see below).

    Parameters
    ----------
    signals : np.ndarray (SIGNALS_DTYPE)

    Returns
    -------
    np.ndarray (np.float32)
    """

    return ffill_na_numba(
        np.where(
            signals["open_long"] > 0,
            1.0,
            np.where(
                signals["open_short"] > 0,
                -1.0,
                np.where(
                    signals["close_long"] > 0,
                    0.0,
                    np.where(signals["close_short"] > 0, 0, np.nan),
                ),
            ),
        )
    )


# ............................. Functions to split signals .............................
@njit
def split_signals(signals: np.ndarray) -> SignalArray3D:  # type: ignore
    """Function to split signals from one- to four-digit_representation.

    This helps to reverse the result from the combine_signals()
    function (see above) which produces a single array with
    singe-digit representataion of signals.

    The function returns the original 3D array with the modified fields
    for 'open_long', 'close_long, etc. modified.

    The intended use is to reconstruct the original signals from the
    one-digit representation if needed for plotting for instance. It
    is defined here so it can be jitted and intended to be used
    through a method of the Signals (wrapper) class.

    Parameters
    ----------
    signals : np.ndarray (SIGNALS_DTYPE)

    Returns
    -------
    np.ndarray (SIGNALS_DTYPE)
    """

    periods, symbols, strategies = signals.shape
    out = np.empty_like(signals, dtype=SIGNALS_DTYPE)

    for k in range(strategies):
        for j in range(symbols):
            position = 0
            for i in range(periods):
                # must be set to 0 individually, as Numba does not support
                # setting all values of the same field at once
                out[i, j, k]["open_long"] = 0
                out[i, j, k]["close_long"] = 0
                out[i, j, k]["open_short"] = 0
                out[i, j, k]["close_short"] = 0

                if signals[i, j, k] > 0:
                    if position != 1:
                        out[i, j, k]["open_long"] = True
                        position = 1
                elif signals[i, j, k] < 0:
                    if position != -1:
                        out[i, j, k]["open_short"] = True
                        position = -1
                else:
                    if position == 1:
                        out[i, j, k]["close_long"] = True
                    elif position == -1:
                        out[i, j, k]["close_short"] = True
                    position = 0

    return out


# .................. Functions to perform math operations on  signals ..................



# ================================= SignalStore JIT Class ==============================
@jitclass(
    [
        ("data", float32[:, :, :]),
    ]
)
class SignalStore:
    """A JIT class for storing and manipulating signals.

    This class should preferrably be used inside
    the Signals wrapper class, which provides methods for convenient
    access to the signals data and makes sure that SignalStore is
    instanitated with the correct data format.

    The .data attribute is expected to receive/hold be a 3D numpy
    array (np.float32). Instances can be added with each other to
    combine the signals form multiple stratagies or parameter sets.

    Why does this class exist? We could also hold the data inside the 
    Signals wrapper class, but the SignalStore class can be used inside
    the BackTestCore class which only accepts JIT classes as arguments.
    """

    def __init__(self, data: npt.ArrayLike):
        self.data = data

    def __add__(self, other):
        if isinstance(other, SignalStore):
            return SignalStore(self.data + other.data)

        elif isinstance(other, (float, int)):
            return SignalStore(np.add(self.data, np.float32(other)))

        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

    def __radd__(self, other):
        return self.__add__(other)

    # @property
    # def open_long(self) -> np.ndarray:
    #     return self.data["open_long"]

    # @property
    # def close_long(self) -> np.ndarray:
    #     return self.data["close_long"]


# ================================= Signals wrapper Class ==============================
class Signals(BaseWrapper3D):
    """Wrapper for the SignalStore class."""

    def __init__(
        self, 
        symbols: list[str], 
        layers: list[str], 
        signals: SignalsArrayT
    ) -> None:

        dtype = np.dtype(signals[0][0][0])
        
        if dtype == SIGNALS_DTYPE:
            self._store = SignalStore(combine_signals(signals))
        elif dtype == np.float32:
            self._store = SignalStore(signals)
        else:
            raise TypeError(f"Unsupported Numpy dtype: '{np.dtype(signals)}'")

        super().__init__(self._store.data, symbols, layers)
        self.symbols = self.columns
        self.strategies = self.layers

    def __add__(self, other) -> "Signals":
        if isinstance(other, Signals):
            if self.symbols != other.symbols:
                raise ValueError("Symbols must match for addition")
            if self.layers != other.layers:
                raise ValueError("Strategies must match for addition")
            new_store = self._store + other._store
        elif isinstance(other, (float, int)):
            new_store = self._store + other
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

        return Signals(symbols=self.symbols, layers=self.layers, signals=new_store.data)

    def __radd__(self, other) -> "Signals":
        return self.__add__(other)

    # ..................................................................................
    @property
    def data(self):
        return self._store.data
