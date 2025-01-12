#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:00:23 2021

@author_ dhaneor
"""

import pytest
from pprint import pprint

from analysis import (
    SignalGenerator,
    SignalGeneratorDefinition,
    signal_generator_factory,
    MarketData,
)
from models.enums import COMPARISON

print("Test file is being executed")


# Fixture for a basic SignalGenerator instance
@pytest.fixture
def basic_signal_generator():
    def_ = SignalGeneratorDefinition(
        name="Test Strategy",
        operands={
            "close": "close", 
            "sma": ("sma", {"timeperiod": 14})
        },
        conditions={
            "open_long": [("close", COMPARISON.CROSSED_ABOVE, "sma")],
            "close_long": [("close", COMPARISON.CROSSED_BELOW, "sma")],
        }
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

def test_subplots_property(basic_signal_generator):
    subplots = basic_signal_generator.subplots
    assert len(subplots) > 0
    assert all(hasattr(subplot, 'name') for subplot in subplots)

def test_randomize_method(basic_signal_generator):
    original_value = basic_signal_generator.parameter_values[0]
    basic_signal_generator.randomize()
    assert basic_signal_generator.parameter_values[0] != original_value

def test_repr_method(basic_signal_generator):
    repr_string = repr(basic_signal_generator)
    assert "Test Strategy" in repr_string
    assert "close" in repr_string
    assert "sma" in repr_string

# Add more tests as needed for other methods and edge cases

if __name__ == "__main__":
    pytest.main([__file__])
