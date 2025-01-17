#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a signals JIT class to store and transform signals.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""
import logging
import numpy as np
import numpy.typing as npt
from numba import njit, float32, from_dtype, types, typeof
from numba.experimental import jitclass

from analysis.dtypes import SIGNALS_DTYPE
from util.proj_types import SignalsArrayT

logger = logging.getLogger("main.signals")

# define Numba type for signal records
SignalRecord = from_dtype(SIGNALS_DTYPE)
SignalArray3D = types.Array(SignalRecord, 3, "C")


# TODO: refactor the spliot_signals function to accept a §D array with 
#       np.flaot32 dtype (instead of SIGNALS_DTYPE).


@njit
def combine_signals(signals: typeof(SignalArray3D)):  # type: ignore
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
def combine_signals_3D(signals: typeof(SignalArray3D)):  # type: ignore
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
def split_signals(signals: typeof(SignalArray3D)):  # type: ignore
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

    for k in range(strategies):
        for j in range(symbols):
            position = 0
            for i in range(periods):
                # must be st to o individually, as Numba does not support
                # setting all values of the same field at once
                signals[i, j, k]["open_long"] = 0
                signals[i, j, k]["close_long"] = 0
                signals[i, j, k]["open_short"] = 0
                signals[i, j, k]["close_short"] = 0
                
                if signals[i, j, k]["combined"] > 0:
                    if position != 1:
                        signals["open_long"][i, j, k] = True
                        position = 1
                elif signals[i, j, k]["combined"] < 0:
                    if position != -1:
                        signals["open_short"][i, j, k] = True
                        position = -1
                else:
                    if position == 1:
                        signals["close_long"][i, j, k] = True
                    elif position == -1:
                        signals["close_short"][i, j, k] = True
                    position = 0

    return signals


# ================================= SignalStore JIT Class ==============================
@jitclass([("data", float32[:, :, :]),])
class SignalStore:
    """A JIT class for storing and manipulating signals.

    This class should preferrably be used inside
    the Signals wrapper class, which provides methos for convenient 
    access to the signals data and makes sure that SignalStore is 
    instanitated with the correct data format.
    
    The .data attribute is expected to receive/hold be a 3D numpy 
    array (np.float32). Instances can be added with each other to
    combine the signals form multiple stratagies or parameter sets.
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
class Signals:
    """Wrapper for the SignalStore class.""" 

    def __init__(self, symbols: list[str], signals: SignalsArrayT) -> None:
        self.symbols = symbols
        self._store = SignalStore(combine_signals(signals))

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

    def __iter__(self):
        return iter(self._store.data)

    def __len__(self) -> int:
        return self._store.data.shape[0]
    
    def __sizeof__(self):
        return object.__sizeof__(self) + self._store.__sizeof__()

    # ..................................................................................
    @property
    def data(self):
        return self._store.data
