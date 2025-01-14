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

from analysis.models.signals import SignalStore, combine_signals


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
            return np.array([next(cycle) for _ in range(periods)]).reshape(periods, 1, 1).astype(np.float32)

        arrays = {key: create_array(pattern) for key, pattern in base_patterns.items()}

        # Tile arrays for multiple symbols and strategies
        for key in arrays:
            arrays[key] = np.tile(arrays[key], (1, num_symbols, num_strategies))

        return arrays

    return _generate


# --------------------------------------------------------------------------------------
def test_combine_signals(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
    )
    expected_result = td["exp"]

    try:
        np.testing.assert_array_equal(combined, expected_result)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(pd.DataFrame(combined.reshape(-1, combined.shape[-1])))
        pytest.fail("Arrays are not equal")


def test_signal_store_instantiation(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
    )

    signal_store = SignalStore(data=combined)

    expected_result = td["exp"]

    assert isinstance(signal_store, SignalStore)
    assert isinstance(signal_store.data, np.ndarray)
    assert signal_store.data.shape == expected_result.shape
    assert np.array_equal(signal_store.data, combined)


def test_signal_store_add(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
    )

    left = SignalStore(data=combined)
    right = SignalStore(data=combined)
    result = left + right

    expected_result = np.add(combined, combined)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected_result.shape
    assert np.array_equal(result.data, expected_result)


def test_signal_store_add_float(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
    )

    left = SignalStore(data=combined)
    right = 5.0
    result = left + right

    expected_result = np.add(combined, right)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected_result.shape
    np.testing.assert_allclose(result.data, expected_result, rtol=1e-5, atol=1e-8)


def test_signal_store_add_int(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["ol"],
        close_long=td["cl"],
        open_short=td["os"],
        close_short=td["cs"]
    )

    left = SignalStore(data=combined)
    right = 5
    result = left + right

    expected_result = np.add(combined, right)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected_result.shape
    np.testing.assert_allclose(result.data, expected_result, rtol=1e-5, atol=1e-8)