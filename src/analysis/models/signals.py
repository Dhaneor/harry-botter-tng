#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a signals JIT class to store and transform signals.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""

import numpy as np
from numba import int8
from numba.experimental import jitclass
from typing import List


spec = [
    ("open_long", int8[:, :, :]),  # (n_periods, n_assets, n_strategies)
    ("close_long", int8[:, :, :]),  # (n_periods, n_assets, n_strategies)
    ("open_short", int8[:, :, :]),  # (n_periods, n_assets, n_strategies)
    ("close_short", int8[:, :, :]),  # (n_periods, n_assets, n_strategies)
    ("data", int8[:, :, :])  # (n_periods, n_assets, n_strategies)
]

@jitclass(spec)
class SignalStore:
    symbols: List[str]

    def __init__(
            self, 
            symbols: List[str], 
            open_long: np.ndarray,
            close_long: np.ndarray, 
            open_short: np.ndarray, 
            close_short: np.ndarray
    ):
        self.symbols = symbols
        self.data = self.combine_signals(open_long, close_long, open_short, close_short)

    def combine_signals(
        self,
        open_long: np.ndarray,
        close_long: np.ndarray,
        open_short: np.ndarray,
        close_short: np.ndarray,
    ):
        n_periods, n_assets, n_strats = open_long.shape
        positions_out = np.zeros((n_periods, n_assets, n_strats), dtype=np.int8)

        for t in range(n_periods):
            for a in range(n_assets):
                for s in range(n_strats):
                    if t == 0:
                        if open_long[t, a, s] == 1:
                            positions_out[t, a, s] = 1
                        elif open_short[t, a, s] == 1:
                            positions_out[t, a, s] = -1
                    else:
                        positions_out[t, a, s] = positions_out[t-1, a, s]

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

    def _combine_signals_loop_old(
            self, 
            open_long: np.ndarray, 
            close_long: np.ndarray, 
            open_short: np.ndarray, 
            close_short: np.ndarray
    ):
        """
        Combine four trading signal arrays into a single positions array with values in {-1, 0, 1},
        including logic to switch directly from long to short (or short to long) on the same time step.

        Parameters
        ----------
        open_long  : np.ndarray, shape (n_periods, n_assets, n_strats), {0,1}
        close_long : np.ndarray, shape (n_periods, n_assets, n_strats), {0,1}
        open_short : np.ndarray, shape (n_periods, n_assets, n_strats), {0,1}
        close_short: np.ndarray, shape (n_periods, n_assets, n_strats), {0,1}

        Returns
        -------
        positions_out : np.ndarray, shape (n_periods, n_assets, n_strats), in {-1,0,1}
                        -1 => short
                        0 => flat
                        1 => long
        """
        n_periods, n_assets, n_strats = open_long.shape

        # Output array
        positions_out = np.zeros((n_periods, n_assets, n_strats), dtype=np.int8)

        # -----------------------------
        # Initialize positions at t = 0
        # -----------------------------
        for a in range(n_assets):
            for s in range(n_strats):
                if open_long[0, a, s] == 1:
                    positions_out[0, a, s] = 1
                elif open_short[0, a, s] == 1:
                    positions_out[0, a, s] = -1
                else:
                    positions_out[0, a, s] = 0

        # -------------------------------------------------------------------
        # For each subsequent time step, copy the previous positions in bulk,
        # then loop over (asset, strategy) to apply logic for close/switch/open
        # -------------------------------------------------------------------
        for t in range(1, n_periods):
            # Copy over the previous step's positions in one slice operation
            positions_out[t] = positions_out[t - 1]

            for a in range(n_assets):
                for s in range(n_strats):
                    pos = positions_out[
                        t, a, s
                    ]  # current position state (carried forward)

                    # 1) Check if we need to close OR switch out of an existing position
                    # ----------------------------------------------------------------
                    if pos == 1:
                        # If we're long, close if close_long=1 or open_short=1 (switch request)
                        if close_long[t, a, s] == 1 or open_short[t, a, s] == 1:
                            pos = 0  # close the long
                    elif pos == -1:
                        # If we're short, close if close_short=1 or open_long=1 (switch request)
                        if close_short[t, a, s] == 1 or open_long[t, a, s] == 1:
                            pos = 0  # close the short

                    # 2) If we're flat now, check if we should open a new position
                    # -------------------------------------------------------------
                    if pos == 0:
                        if open_long[t, a, s] == 1:
                            pos = 1
                        elif open_short[t, a, s] == 1:
                            pos = -1

                    positions_out[t, a, s] = pos

        return positions_out


if __name__ == "__main__":
    signal_store = SignalStore(["AAPL", "GOOGL", "MSFT"])
    print(signal_store.symbols)  # Output: ['AAPL', 'GOOGL', 'MSFT']
