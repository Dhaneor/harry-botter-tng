#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:00:23 2021

@author_ dhaneor
"""

import numpy as np
import pandas as pd
import pytest
import talib
from pprint import pprint

from analysis import (
    SignalGenerator,
    SignalGeneratorDefinition,
    signal_generator_factory,
    MarketData, 
    MarketDataStore
)
from models.enums import COMPARISON


# Fixture for a basic SignalGenerator instance
@pytest.fixture
def basic_signal_generator():
    def_ = SignalGeneratorDefinition(
        name="Test Strategy",
        operands={"close": "close", "sma": ("sma", {"timeperiod": 14})},
        conditions={
            "open_long": [("close", COMPARISON.CROSSED_ABOVE, "sma")],
            "close_long": [("close", COMPARISON.CROSSED_BELOW, "sma")],
        },
    )
    return signal_generator_factory(def_)


def test_signal_generator_init(basic_signal_generator):
    assert isinstance(basic_signal_generator, SignalGenerator)
    assert basic_signal_generator.name == "Test Strategy"
    assert (
        len(basic_signal_generator.operands) == 2
    ), f"Incorrect number of operands: {basic_signal_generator.operands}"
    assert len(basic_signal_generator.conditions) == 2


def test_indicators_property(basic_signal_generator):
    pprint(basic_signal_generator.__dict__)
    indicators = basic_signal_generator.indicators
    assert len(indicators) == 1
    assert indicators[0].name.lower() == "sma"


def test_parameters_property(basic_signal_generator):
    parameters = basic_signal_generator.parameters
    assert len(parameters) == 1
    assert parameters[0].name == "timeperiod"
    assert parameters[0].value == 14


def test_parameter_values_property(basic_signal_generator):
    assert basic_signal_generator.parameter_values == (14,)


def test_parameter_values_setter(basic_signal_generator):
    basic_signal_generator.parameter_values = (20,)
    assert basic_signal_generator.parameter_values == (20,)
    assert basic_signal_generator.parameters[0].value == 20


def test_market_data_property(basic_signal_generator):
    assert basic_signal_generator.market_data is None


def test_market_data_setter(basic_signal_generator):
    mock_data = MarketData.from_random(length=30, no_of_symbols=1)
    basic_signal_generator.market_data = mock_data
    assert basic_signal_generator.market_data is mock_data


# def test_subplots_property(basic_signal_generator):
#     subplots = basic_signal_generator.subplots
#     assert len(subplots) > 0
#     assert all(hasattr(subplot, 'name') for subplot in subplots)


def test_randomize_method(basic_signal_generator):
    original_value = basic_signal_generator.parameter_values[0]
    basic_signal_generator.randomize()
    assert basic_signal_generator.parameter_values[0] != original_value


def test_repr_method(basic_signal_generator):
    repr_string = repr(basic_signal_generator)
    assert "Test Strategy" in repr_string
    assert "close" in repr_string
    assert "sma" in repr_string


# ........................... TEST EXECUTE METHOD: 1 SYMBOL ............................
@pytest.fixture
def simple_signal_generator_with_data():
    # Define the signal generator using the factory
    def_ = SignalGeneratorDefinition(
        name="Simple Test Strategy",
        operands={"close": "close", "threshold": ("trigger", 10.0, [9, 11, 1])},
        conditions={
            "open_long": [("close", COMPARISON.IS_ABOVE, "threshold")],
            "close_long": [("close", COMPARISON.IS_BELOW, "threshold")],
        },
    )
    sg = signal_generator_factory(def_)
    
    # Create a simple, predictable dataset
    dates = pd.date_range(start='2020-01-01', periods=5)
    data = np.array([
        [8, 9, 7, 8, 1000],  # No signal
        [11, 12, 10, 11, 1000],  # Open long
        [12, 13, 11, 12, 1000],  # Hold long
        [9, 10, 8, 9, 1000],  # Close long
        [8, 9, 7, 8, 1000],  # No signal
    ], dtype=np.float32)
    
    # Create MarketDataStore instance
    mds = MarketDataStore(
        timestamp=dates.astype(np.int64).values.reshape(-1, 1),
        open_=data[:, 0].reshape(-1, 1),
        high=data[:, 1].reshape(-1, 1),
        low=data[:, 2].reshape(-1, 1),
        close=data[:, 3].reshape(-1, 1),
        volume=data[:, 4].reshape(-1, 1)
    )
    
    # Create MarketData instance
    symbols = ['TESTSYMBOL']
    md = MarketData(mds, symbols)
    
    # Set the market data for the signal generator
    sg.market_data = md
    
    return sg

def test_execute_returns_correct_format(simple_signal_generator_with_data):
    sg = simple_signal_generator_with_data
    result = sg.execute()
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert set(result.keys()) == {'open_long', 'open_short', 'close_long', 'close_short'}, \
        "Result should have keys 'open_long', 'open_short', 'close_long', 'close_short'"
    
    for key in ('open_long', 'close_long'):
        value = result[key]

        assert isinstance(value, np.ndarray), f"{key} should be a numpy array"
        assert value.ndim == 3, f"{key} should be a 3D array"
        assert value.shape[0] == len(sg.market_data), f"{key} should have same length as market data"
        assert value.shape[1] == 1, f"{key} should have 1 column"
        assert value.shape[2] == 1, f"{key} should have depth 1"
        assert value.dtype == bool, f"{key} should contain boolean values"


def test_execute_returns_correct_data(simple_signal_generator_with_data):
    sg = simple_signal_generator_with_data
    result = sg.execute()

    assert result["open_short"] is None, "open_short should be None"
    assert result["close_short"] is None, "close_short should be None"
    
    # Check the shape of the result
    for k,v in result.items():
        if k not in ('open_long', 'close_long'):
            assert v is None
            continue

        assert v.shape == (5, 1, 1), \
            f"[{k}] Incorrect shape of the result array: {v} ({v.shape})"
                
        if k == 'open_long':
            signals = v[:, 0, 0]  # open_long signals
            np.testing.assert_array_equal(
                signals,
                [0, 1, 1, 0, 0],
                "Incorrect open_long signals: {signals}"
            )
        if k == 'close_long':
            signals = v[:, 0, 0]
            np.testing.assert_array_equal(
                signals,
                [1, 0, 0, 1, 1],
                "Incorrect close_long signals: {signals}"
            )

# ...................... TEST EXECUTE METHOD: MUTLIPLE SYMBOLs .........................
@pytest.fixture
def generate_test_data():
    def _generate(num_symbols=1, periods=5):
        dates = pd.date_range(start='2020-01-01', periods=periods)
        
        # Generate random data for the specified number of periods
        data = np.random.randint(7, 14, size=(periods, 5)).astype(np.float32)
        data[:, -1] = 1000  # Set volume to 1000 for all periods
        
        # Repeat the data for each symbol
        data = np.tile(data, (num_symbols, 1, 1))
        
        mds = MarketDataStore(
            timestamp=dates.astype(np.int64).values.reshape(-1, 1),
            open_=data[:, :, 0].T,
            high=data[:, :, 1].T,
            low=data[:, :, 2].T,
            close=data[:, :, 3].T,
            volume=data[:, :, 4].T
        )
        
        symbols = [f'SYMBOL{i+1}' for i in range(num_symbols)]
        md = MarketData(mds, symbols)
        
        return md
    
    return _generate

@pytest.fixture
def calculate_expected_result():
    def _calculate(market_data):
        close_data = market_data.close
        threshold = 10.0
        
        open_long = (close_data > threshold) 
        close_long = (close_data < threshold) 

        return {
            'open_long': open_long[:, :, np.newaxis],
            'close_long': close_long[:, :, np.newaxis],
            'open_short': None,
            'close_short': None
        }
    
    return _calculate

@pytest.fixture
def simple_signal_generator():
    def _create_sg(market_data):
        def_ = SignalGeneratorDefinition(
            name="Simple Test Strategy",
            operands={"close": "close", "threshold": ("trigger", 10.0, [9, 11, 1])},
            conditions={
                "open_long": [("close", COMPARISON.IS_ABOVE, "threshold")],
                "close_long": [("close", COMPARISON.IS_BELOW, "threshold")],
            },
        )
        sg = signal_generator_factory(def_)
        sg.market_data = market_data
        return sg
    
    return _create_sg

@pytest.mark.parametrize("num_symbols", [1, 2])
def test_execute_returns_correct_data_ext(generate_test_data, calculate_expected_result, simple_signal_generator, num_symbols):
    market_data = generate_test_data(num_symbols)
    sg = simple_signal_generator(market_data)
    result = sg.execute()
    expected_result = calculate_expected_result(market_data)

    print(num_symbols)
    print(market_data.close)
    print(result)
    
    for key in ('open_long', 'close_long', 'open_short', 'close_short'):
        if expected_result[key] is None:
            assert result[key] is None, f"{key} should be None"
        else:
            assert np.array_equal(result[key], expected_result[key]), \
                f"Incorrect {key} signals. expected: {expected_result[key]}, actual: {result[key]} for {key}"
        
        if result[key] is not None:
            assert result[key].shape == (5, num_symbols, 1), f"Incorrect shape for {key}"


# ..................... TEST EXECUTE METHOD: COMPLEX CONDITION .........................
@pytest.fixture
def complex_signal_generator():
    def _create_sg(market_data):
        def_ = SignalGeneratorDefinition(
            name="Complex Test Strategy",
            operands={
                "close": "close",
                "sma": ("sma", {"timeperiod": 2}),
                "threshold": ("trigger", 10.0, [9, 11, 1]),
            },
            conditions={
                "open_long": [
                    [
                        ("close", COMPARISON.IS_ABOVE, "threshold"),
                        ("close", COMPARISON.CROSSED_ABOVE, "sma"),
                    ]
                ],
                "close_long": [
                    [
                        ("close", COMPARISON.IS_BELOW, "threshold"),
                        ("close", COMPARISON.CROSSED_BELOW, "sma"),
                    ]
                ],
            },
        )
        sg = signal_generator_factory(def_)
        sg.market_data = market_data
        return sg
    
    return _create_sg


@pytest.fixture
def calculate_complex_expected_result():
    def _calculate(market_data):
        close_data = market_data.close
        threshold = 10.0
        
        # Calculate SMA for each symbol
        sma = np.zeros_like(close_data)
        for i in range(close_data.shape[1]):
            sma[:, i] = talib.SMA(close_data[:, i].astype(np.float64), timeperiod=2)
        
        above_threshold = (close_data > threshold)
        below_threshold = (close_data < threshold)
        crossed_above_sma = (close_data > sma) & (np.roll(close_data, 1, axis=0) <= np.roll(sma, 1, axis=0))
        crossed_below_sma = (close_data < sma) & (np.roll(close_data, 1, axis=0) >= np.roll(sma, 1, axis=0))
        
        open_long = above_threshold & crossed_above_sma
        close_long = below_threshold & crossed_below_sma

        # Set the first row to False as we can't determine crosses for it
        open_long[0, :] = False
        close_long[0, :] = False

        return {
            'open_long': open_long[:, :, np.newaxis],
            'close_long': close_long[:, :, np.newaxis],
            'open_short': None,
            'close_short': None
        }
    
    return _calculate


@pytest.mark.parametrize("num_symbols", [1, 2])
def test_execute_returns_correct_data_complex(generate_test_data, calculate_complex_expected_result, complex_signal_generator, num_symbols):
    # Generate market data with 10 periods
    market_data = generate_test_data(num_symbols, periods=10)
    
    # Create the signal generator with the generated market data
    sg = complex_signal_generator(market_data)
    
    # Execute the signal generator
    result = sg.execute()
    
    # Calculate the expected result using the same market data
    expected_result = calculate_complex_expected_result(market_data)

    for key in ('open_long', 'close_long', 'open_short', 'close_short'):
        if result[key] is not None:
            assert result[key].shape == (10, num_symbols, 1), f"Incorrect shape for {key}"

        if expected_result[key] is None:
            assert result[key] is None, f"{key} should be None"
        else:
            try:
                np.testing.assert_array_equal(result[key], expected_result[key], 
                    f"Incorrect {key} signals.")
            except AssertionError:
                print(f"\nMismatch in {key} signals:")
                print(f"Market Data (Close):\n{market_data.close}")
                print(f"\nExpected {key}:\n{expected_result[key].squeeze()}")
                print(f"\nActual {key}:\n{result[key].squeeze()}")
                print("\nDifferences:")
                diff = result[key] != expected_result[key]
                for i in range(diff.shape[0]):
                    for j in range(diff.shape[1]):
                        if diff[i, j]:
                            print(f"Mismatch at index [{i}, {j}]: Expected {expected_result[key][i, j, 0]}, Got {result[key][i, j, 0]}")

                # Additional debugging information
                print("\nThreshold:", sg.operands['threshold'])
                print("\nSMA values:")
                for i in range(num_symbols):
                    print(f"Symbol {i+1}: {talib.SMA(market_data.close[:, i].astype(np.float64), timeperiod=2)}")

                # for operand in sg.operands.values():
                #     print(f"\nOperand {operand} cache: {operand._cache}")

                raise AssertionError("Test failed due to mismatches in signal generation.")


if __name__ == "__main__":
    pytest.main([__file__])
