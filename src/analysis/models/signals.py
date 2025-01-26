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
from numba import njit, float32, boolean, from_dtype, types, typeof
from numba.experimental import jitclass

from analysis.dtypes import SIGNALS_DTYPE
from util.proj_types import SignalsArrayT
from misc.base_wrapper import BaseWrapper3D
from misc.numba_funcs import ffill_na_numba

logger = logging.getLogger("main.signals")

# define Numba type for signal records
SignalRecord = from_dtype(SIGNALS_DTYPE)
SignalArray3D = types.Array(SignalRecord, 3, "C")


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
@njit
def normalize_signal(signal, lookback_period=20, clip=2, weighted=True, alpha=None):
    """
    Normalizes a signal using a rolling lookback period with exponential weighting.

    Args:
        signal (np.ndarray): The input signal as a 1D NumPy array.
        lookback_period (int): The lookback period for calculating the mean.
        clip (float): The maximum absolute value of the normalized signal.
        alpha (float): Decay factor for exponential weighting (optional). If None, defaults to 2 / (lookback_period + 1).

    Returns:
        np.ndarray: The normalized signal.
    """
    if weighted:
      if alpha is None:
          alpha = 2 / (lookback_period + 1)  # Default decay factor
      
      weights = np.array([(1 - alpha) ** i for i in range(lookback_period)][::-1])
      weights /= np.sum(weights)  # Normalize weights to sum to 1
    else:
      weights = np.ones(lookback_period+1)

    normalized_signal = np.zeros_like(signal, dtype=np.float64)

    for i in range(len(signal)):
        start_index = max(0, i - lookback_period + 1)
        lookback_data = np.abs(signal[start_index:i + 1])
        
        weighted_data = lookback_data * weights[-len(lookback_data):]
        weighted_mean = np.sum(weighted_data) if weighted else np.mean(weighted_data)
        scaling_factor = 1 / weighted_mean if weighted_mean != 0 else 0
        
        normalized_signal[i] = signal[i] * scaling_factor
  
    return np.clip(normalized_signal, -clip, clip)



# ================================= SignalStore JIT Class ==============================
@jitclass(
    [
        ("data", float32[:, :, :]),
        ("is_normalized", boolean),
    ]
)
class SignalStore:
    """A JIT class for storing and manipulating signals.

    This class should preferrably be used inside
    the Signals wrapper class, which provides methods for convenient
    access to the signals data and makes sure that SignalStore is
    instantiated with the correct data format.

    The .data attribute is expected to receive/hold be a 3D numpy
    array (np.float32). Instances can be added with each other to
    combine the signals form multiple stratagies or parameter sets.

    Why does this class exist? We could also hold the data inside the 
    Signals wrapper class, but the SignalStore class can be used inside
    the BackTestCore class which only accepts JIT classes as arguments.
    """

    def __init__(self, data: npt.ArrayLike):
        self.data = data
        self.is_normalized = False

    def __add__(self, other):
        if isinstance(other, SignalStore):
            return SignalStore(self.data + other.data)

        elif isinstance(other, (float, int)):
            return SignalStore(np.add(self.data, np.float32(other)))

        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

    def __radd__(self, other):
        return self.__add__(other)

    def combine(self, normalize: bool = False) -> None:
        """Combines signals from all strategies/parameter sets.
        
        Takes the signals from all strategies/parameter sets for each
        symbol and combines them into one signal, which can then be used
        by subsequent components.

        When to use combined signals? This is useful for live/paper
        trading, but should not be used for optimisation runs.

        Should only be done at the sub-portfolio level, after all signals 
        from different strategies have been collected and added with each 
        other.

        Parameters
        ----------
        normalize : bool, optional
            Determines whether to normalize the combined signals, see 
            normalize() method below for more details.

        Returns:
        --------
        None
            Just changes the internal data attribute.
        """
        self.data = np.sum(self.data, axis=2)

        if normalize:
            self.normalize()

    def _normalize(self, period: int = 20) -> None:
        """Normalizes the combined signals.

        NOTE:
        This method only makes sense for signals that have been 
        combined/added (per symbol/market). Clients should use the
        combine method and set the normalize parameter to True.

        This is inspired / taken from Robert Carver’s book "Sytematic
        Trading".
        
        - Calculate the “raw” forecast or signal for neach day/period.
        - Measure the average absolute size of these raw signals over 
        a chosen lookback window. For example, over the last 20 days, 
        compute the average of the absolute values of the signals.
        - Compute a scaling factor so that the average signal ends 
        up at the “target” (Carver often uses 10, but we will use 1).
        - If S is the average absolute signal over the lookback, then
        scaling factor = 10 / S
        - Apply the scaling factor to the raw signal to get the final 
        signal. That is: final signal} = raw signal x scaling factor

        In other words, if the average absolute signal is less than 10, 
        you scale up so the signals average out to 10; if the average 
        is above 10, you scale down. This keeps the typical forecast 
        (and hence your typical position size or weighting) in a 
        relatively stable zone.

	    Why this helps:
		- It stabilizes the typical signal level, so that day-to-day 
        fluctuations in your system’s raw forecasts don’t cause wild 
        changes in position sizing. It ensures that your forecast’s 
        long-term average magnitude aligns with the desired “unit 
        exposure” (which Carver picks as 10).

        Parameters:
        -----------
        period : int, optional
            The lookback period for computing the average absolute signal.

        Returns:
        --------
        None
            Just changes the internal data attribute. The array will 
            still have  three dimensions to make it compatible with
            other components, the shape is (periods, symbols, 1).
        """

        # we need to make sure that the signals from multiple strategies
        # have already been combined (per symbol/market)
        if self.data.shape[2] > 1:
            self.combine(normalize=False)

        data = self.data
        periods, markets, _ = data.shape

        out = np.empty(shape=(periods, markets), dtype=np.float32)

        for p in range(periods):
            if p < period:
                out[p, :] = data[p, :]
            else:
                for m in range(markets):
                    scaling_factor = 1 / np.mean(np.abs(data[p - period:p, m]))
                    out[p, m] = data[p, m] * scaling_factor

        del data
        self.data = out
        self.is_normalized = True


# ================================= Signals wrapper Class ==============================
class Signals(BaseWrapper3D):
    """Wrapper for the SignalStore class."""

    def __init__(
        self, 
        symbols: list[str], 
        layers: list[str], 
        signals: np.ndarray | SignalsArrayT
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
            new = self._store.data + other._store.data
        elif isinstance(other, (float, int)):
            new = self._store + other
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other)}'")

        return Signals(symbols=self.symbols, layers=self.layers, signals=new)

    def __radd__(self, other) -> "Signals":
        return self.__add__(other)

    # ..................................................................................
    @property
    def data(self):
        return self._store.data
    
    # ..................................................................................
    def apply_weight(self, weight: float) -> "Signals":
        return Signals(
            symbols=self.symbols,
            layers=self.layers,
            signals=self._store.data * weight
        )
    
    def normalize(self, lookback: int = 30) -> "Signals":
        """Normalizes the signals by subtracting the rolling mean and dividing by the rolling standard deviation."""


# ==================== Functions for convenient testing of Signals =====================
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


def get_signals_instance():
    """Returns an instance of the Signals class.

    This function is intended to be used inside the BacktestCore class
    to create a Signals instance. It is designed to be a factory function
    that returns a Signals instance for testing purposes.  
    """
    td = generate_test_data(100, 10, 10)
    symbols = [f"Symbol_{i}" for i in range(td.shape[1])]
    layers = [f"Layer_{i}" for i in range(td.shape[2])]

    return Signals(symbols=symbols, layers=layers, signals=td)   
