#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides tests for the SignalStore class (pytest).

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""
import itertools
import numpy as np
import pandas as pd
import pytest

from numba.typed import List as NumbaList

from analysis.models.signals import SignalStore


@pytest.fixture
def generate_test_data():
    def _generate(periods=8, num_symbols=1, num_strategies=1):
        base_patterns = {
            "ol": (1, 0, 0, 0, 1, 0, 0, 0),
            "cl": (0, 1, 0, 0, 0, 0, 0, 0),
            "os": (0, 0, 1, 0, 0, 0, 1, 0),
            "cs": (0, 0, 0, 1, 0, 0, 0, 0),
            "exp": (1, 0, -1, 0, 1, 1, -1, -1),
        }

        def create_array(pattern):
            cycle = itertools.cycle(pattern)
            return np.array([next(cycle) for _ in range(periods)]).reshape(periods, 1, 1)

        arrays = {key: create_array(pattern) for key, pattern in base_patterns.items()}

        # Tile arrays for multiple symbols and strategies
        for key in arrays:
            arrays[key] = np.tile(arrays[key], (1, num_symbols, num_strategies))

        return arrays

    return _generate


# def test_signal_store():
#     symbols = NumbaList()
#     symbols.append('AAPL')
#     symbols.append('GOOGL')
#     symbols.append('MSFT')
#     signal_store = SignalStore(symbols)
#     assert signal_store.symbols == symbols


# def test_signal_store_with_direct_list_init():
#     symbols = NumbaList(['AAPL', 'GOOGL', 'MSFT'])
#     signal_store = SignalStore(symbols)
#     assert signal_store.symbols == symbols


def test_instantiation(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    signal_store = SignalStore(
        symbols=NumbaList(['AAPL', 'GOOGL', 'MSFT']),
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
        )

    expected_result = td["exp"]

    try:
        np.testing.assert_array_equal(signal_store.data, expected_result)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        df = pd.DataFrame(signal_store.data.reshape(-1, signal_store.data.shape[-1]))
        print(df)
        pytest.fail("Arrays are not equal")