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

from analysis.models.signals import (  # noqa: F401
    Signals, SignalStore, 
    combine_signals, combine_signals_np, 
    split_signals
)
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

    expected = np.array(td["combined"].copy())  # Convert to float32 for testing)
    actual = combine_signals(td)

    assert expected.shape == actual.shape, "Shape mismatch"

    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(pd.DataFrame(actual.reshape(-1, actual.shape[-1])))
        pytest.fail("Arrays are not equal")
    except Exception as e:
        print(f"Error: {e}")
        print(np.info(actual))
        print(np.info(expected))
        pytest.fail("Unexpected error")


def test_combine_signals_np(generate_test_data):
    td = generate_test_data(periods=10, num_symbols=1, num_strategies=1)

    expected = np.array(td["combined"].copy())  # Convert to float32 for testing)
    actual = combine_signals_np(td)

    assert expected.shape == actual.shape, "Shape mismatch"

    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(pd.DataFrame(actual.reshape(-1, actual.shape[-1])))
        pytest.fail("Arrays are not equal")
    except Exception as e:
        print(f"Error: {e}")
        print(np.info(actual))
        print(np.info(expected))
        pytest.fail("Unexpected error")


def test_split_sginals(generate_test_data):
    td = generate_test_data(periods=10, num_symbols=1, num_strategies=1)

    result = split_signals(td["combined"])
            
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


# ------------------------- TESTS FOR SignalStore class --------------------------------
def test_signal_store_instantiation(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)
    data = combine_signals(td)
    signal_store = SignalStore(data=data)

    assert isinstance(signal_store, SignalStore)
    assert isinstance(signal_store.data, np.ndarray)
    assert signal_store.data.shape == td.shape
    assert np.array_equal(signal_store.data, data)


def test_signal_store_add(generate_test_data):
    td = generate_test_data(periods=100, num_symbols=2, num_strategies=2)

    data = combine_signals(td)

    left = SignalStore(data=data)
    right = SignalStore(data=data)
    result = left + right

    expected_result = np.add(data, data)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected_result.shape
    assert np.array_equal(result.data, expected_result)


def test_signal_store_add_float(generate_test_data):
    td = generate_test_data(periods=100, num_symbols=2, num_strategies=2)

    left = SignalStore(data=combine_signals(td))
    right = 5.0
    result = left + right

    expected = combine_signals(td.copy())
    assert isinstance(expected, np.ndarray)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected.shape
    # np.testing.assert_allclose(result.data, expected, rtol=1e-5, atol=1e-8)


def test_signal_store_add_int(generate_test_data):
    td = generate_test_data(periods=100, num_symbols=2, num_strategies=5)

    data = combine_signals(td)
    left = SignalStore(data=data)
    right = int(5)
    result = left + right

    expected = np.add(data, right)
    assert isinstance(expected, np.ndarray)

    assert isinstance(result, SignalStore)
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == expected.shape
    # np.testing.assert_allclose(result.data, expected, rtol=1e-5, atol=1e-8)


# --------------------------- TESTS FOR Signals class ----------------------------------
def test_signals_instantiation(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)

    symbols = [f"Symbol_{i}" for i in range(td.shape[1])]
    layers = [f"Layer_{i}" for i in range(td.shape[2])]

    signals = Signals(symbols, layers, td)

    assert isinstance(signals, Signals)
    assert isinstance(signals.data, np.ndarray)
    assert signals.data.shape == td.shape
    assert np.array_equal(signals.data, combine_signals(td))

def test_signals_add(generate_test_data):
    td = generate_test_data(periods=1000, num_symbols=1, num_strategies=1)
    data = combine_signals(td)

    symbols = [f"Symbol_{i}" for i in range(td.shape[1])]
    layers = [f"Layer_{i}" for i in range(td.shape[2])]

    left = Signals(symbols, layers, td)
    right = Signals(symbols, layers, td)
    signals = left + right

    assert isinstance(signals, Signals)
    assert isinstance(signals.data, np.ndarray)
    assert signals.data.shape == td.shape
    assert np.array_equal(signals.data, np.multiply(2, data))