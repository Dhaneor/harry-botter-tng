#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import operator
import pytest
import sys
from functools import reduce

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from indicators.indicator import (  # noqa: F401, E402
    factory,
    IIndicator,
    PlotDescription,
)

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "indicator_name,params,source,expected_name",
    [
        ("SMA", {"timeperiod": 14}, "talib", "SMA"),
        ("EMA", {"timeperiod": 14}, "talib", "EMA"),
        (
            "BBANDS",
            {"matype": 0, "timeperiod": 5, "nbdevup": 2, "nbdevdn": 2},
            "talib",
            "BBANDS",
        ),
        ("ADX", {"timeperiod": 14}, "talib", "ADX"),
        (
            "MACD",
            {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            "talib",
            "MACD",
        ),
        (
            "rsi_overbought",
            {"rsi_overbought": 70, "parameter_space": {"rsi_overbought": [60, 90, 3]}},
            "fixed",
            "rsi_overbought",
        ),
    ],
)
def test_indicator_factory(indicator_name, params, source, expected_name):
    indicator = factory(indicator_name, params=params, source=source)
    logger.info(indicator.__dict__)

    if source == "fixed":
        del params["parameter_space"]

    assert indicator.name == expected_name.lower()
    assert isinstance(indicator, IIndicator)
    assert {p.name: p.value for p in indicator.parameters} == params


@pytest.mark.parametrize(
    "indicator_name,params,source,unique_name",
    [
        ("SMA", {"timeperiod": 14}, "talib", "sma_14"),
        ("EMA", {"timeperiod": 14}, "talib", "ema_14"),
        (
            "BBANDS",
            {"matype": 0, "timeperiod": 5, "nbdevup": 2, "nbdevdn": 2.2},
            "talib",
            "bbands_5_2_2.2_0",
        ),
        ("ADX", {"timeperiod": 14}, "talib", "adx_14"),
        (
            "MACD",
            {"fastperiod": 12, "signalperiod": 9, "slowperiod": 26},
            "talib",
            "macd_12_26_9",
        ),
        (
            "rsi_overbought",
            {"rsi_overbought": 70, "parameter_space": {"rsi_overbought": [60, 90, 3]}},
            "fixed",
            "rsi_overbought_70",
        ),
    ],
)
def test_indicator_properties(indicator_name, params, source, unique_name):
    logger.info(f"Testing {indicator_name} with params: {params} and source: {source}")
    indicator = factory(indicator_name, params=params, source=source)

    if source == "fixed":
        del params["parameter_space"]

    # check the different names
    assert indicator.name == indicator_name.lower()
    assert indicator.unique_name == unique_name
    assert isinstance(indicator.display_name, str)

    # check if an output (tuple) is returned
    assert isinstance(indicator.unique_output, tuple)
    assert isinstance(indicator.unique_output[0], str)

    # check if valid_params (tuple) is returned
    assert isinstance(indicator.valid_params, tuple)
    assert isinstance(indicator.valid_params[0], str)

    # check if parameters were set and a (dict) is returned
    assert {p.name: p.value for p in indicator.parameters} == params
    assert isinstance(indicator.parameters, tuple)

    # check if parameter_space (dict) is returned
    assert isinstance(indicator.parameter_space, dict)
    # assert isinstance(indicator.parameter_space.get("timeperiod"), Parameter)

    # check if plot_desc (PlotDescription) is returned
    assert isinstance(indicator.plot_desc, PlotDescription)


@pytest.mark.parametrize(
    "indicator_name,params,source,expected_exception",
    [
        (
            "SMA",
            {"timeperiod": 0},
            "talib",
            ValueError,
        ),  # Example of an invalid timeperiod
        (
            "SMA",
            {"timeperiod": 1000},
            "talib",
            ValueError,
        ),  # Example of an invalid timeperiod
        (
            "SMA",
            {"timeperiod": -5},
            "talib",
            ValueError,
        ),  # Example of an invalid timeperiod
        (
            "EMA",
            {"timeperiod": "invalid"},
            "talib",
            TypeError,
        ),  # Example of a wrong type
        (
            "BBANDS",
            {"matype": 0, "timeperiod": 5, "nbdevup": "invalid", "nbdevdn": 2},
            "talib",
            TypeError,
        ),
        ("ADX", {"timeperiod": None}, "talib", TypeError),  # Example of a None value
        (
            "MACD",
            {"fastperiod": 12, "slowperiod": 26, "signalperiod": "invalid"},
            "talib",
            TypeError,
        ),
    ],
)
def test_indicator_factory_invalid_values(
    indicator_name, params, source, expected_exception
):
    logger.debug("Testing %s with invalid values: %s ", indicator_name, params)
    with pytest.raises(expected_exception):
        factory(indicator_name, params=params, source=source)
        logger.error(f"Test case for {indicator_name} failed.")


@pytest.mark.parametrize(
    "indicator_name,params,source,expected_name",
    [
        ("SMA", {"timeperiod": 14}, "talib", "SMA"),
        ("EMA", {"timeperiod": 14}, "talib", "EMA"),
        (
            "BBANDS",
            {"matype": 0, "timeperiod": 5, "nbdevup": 2, "nbdevdn": 2},
            "talib",
            "BBANDS",
        ),
        ("ADX", {"timeperiod": 14}, "talib", "ADX"),
        (
            "MACD",
            {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            "talib",
            "MACD",
        ),
        (
            "rsi_overbought",
            {"rsi_overbought": 70, "parameter_space": {"rsi_overbought": [60, 90, 3]}},
            "fixed",
            "rsi_overbought",
        ),
    ],
)
def test_parameter_combinations(indicator_name, params, source, expected_name):
    logger.debug("Testing %s with parameter combinations: %s ", indicator_name, params)
    indicator = factory(indicator_name, params=params, source=source)
    assert indicator.name == expected_name.lower()

    combinations = tuple(indicator.parameter_combinations)
    logger.info(combinations)

    lst = [sum(1 for _ in p) for p in indicator.parameters]
    assert len(set(combinations)) == reduce(operator.mul, lst, 1)
