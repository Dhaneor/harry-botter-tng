#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:00:23 2021

@author_ dhaneor
"""

import numpy as np
import pandas as pd
import pytest
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


# ............................... TEST EXECUTE METHOD .................................
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


if __name__ == "__main__":
    pytest.main([__file__])
