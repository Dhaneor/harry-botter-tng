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

from analysis.models.signals import SignalStore, combine_signals, split_signals
from analysis import SIGNALS_DTYPE
from util import get_logger

logger = get_logger("main")


@pytest.fixture
def generate_test_data():
    def _generate(periods=8, num_symbols=1, num_strategies=1) -> np.ndarray:
        base_patterns = {
            "open_long":   (1, 0,  0, 0, 1, 0,  0,  0),
            "close_long":  (0, 1,  0, 0, 0, 0,  0,  0),
            "open_short":  (0, 0,  1, 0, 0, 0,  1,  0),
            "close_short": (0, 0,  0, 1, 0, 0,  0,  0),
            "combined":    (1, 0, -1, 0, 1, 1, -1, -1),
        }

        def create_array(pattern):
            cycle = itertools.cycle(pattern)
            return np.array([next(cycle) for _ in range(periods)]).reshape(periods, 1, 1).astype(np.float32)
        
        out = np.empty((periods, num_symbols, num_strategies), dtype=SIGNALS_DTYPE)
        
        for key in base_patterns.keys():
            out[key] = create_array(base_patterns[key])

        return np.tile(out, (1, num_symbols, num_strategies))

    return _generate


# --------------------------------------------------------------------------------------
def test_combine_signals(generate_test_data):
    td = generate_test_data(periods=10, num_symbols=1, num_strategies=1)

    expected_result = np.array(td["combined"])

    combined = combine_signals(
        open_long=td["open_long"],
        close_long=td["close_long"],
        open_short=td["open_short"],
        close_short=td["close_short"]
    )

    try:
        np.testing.assert_array_equal(combined, expected_result)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(pd.DataFrame(combined.reshape(-1, combined.shape[-1])))
        pytest.fail("Arrays are not equal")


def test_split_sginals(generate_test_data):
    td = generate_test_data(periods=10, num_symbols=1, num_strategies=1)

    combined = td["combined"]
    result = split_signals(combined)

    result = dict(zip(("open_long", "close_long", "open_short", "close_short"), result))
            
    for i in range(td.shape[0]):
        for j in range(td.shape[1]):
            for k in range(td.shape[2]):
                ol = td["open_long"][i, j, k]
                cl = td["close_long"][i, j, k]
                os = td["open_short"][i, j, k]
                cs = td["close_short"][i, j, k]

                aol = result["open_long"][i, j, k]
                acl = result["close_long"][i, j, k]
                aos = result["open_short"][i, j, k]
                acs = result["close_short"][i, j, k]
                
                try:
                    assert ol == aol and cl == acl and os == aos and cs == acs, \
                        f"Mismatch at index [{i}, {j}, {k}]"
                except AssertionError as e:
                    print(f"Assertion Error: {e}")
                    # print(f"{ol} {cl} {os} {cs} -> {aol} {acl} {aos} {acs}")
                    print()
                    pytest.fail("Arrays are not equal")



def test_signal_store_instantiation(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["open_long"],
        close_long=td["close_long"],
        open_short=td["open_short"],
        close_short=td["close_short"]
    )

    signal_store = SignalStore(data=combined)

    expected_result = td["combined"]

    assert isinstance(signal_store, SignalStore)
    assert isinstance(signal_store.data, np.ndarray)
    assert signal_store.data.shape == expected_result.shape
    assert np.array_equal(signal_store.data, combined)


def test_signal_store_add(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    combined = combine_signals(
        open_long=td["open_long"],
        close_long=td["close_long"],
        open_short=td["open_short"],
        close_short=td["close_short"]
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
        open_long=td["open_long"],
        close_long=td["close_long"],
        open_short=td["open_short"],
        close_short=td["close_short"]
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
        open_long=td["open_long"],
        close_long=td["close_long"],
        open_short=td["open_short"],
        close_short=td["close_short"]
    )

    left = SignalStore(data=combined)
    right = 5
    result = left + right

    expected_result = np.add(combined, right)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected_result.shape
    np.testing.assert_allclose(result.data, expected_result, rtol=1e-5, atol=1e-8)
