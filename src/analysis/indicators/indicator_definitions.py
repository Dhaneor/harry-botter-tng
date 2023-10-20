#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the indicator definitions that are used by IndicatorFactory
to build the indicators.

Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import enum
import talib as ta
from typing import Dict, List, Optional, NamedTuple, Tuple, Union, Callable

@enum.global_enum
class IndicatorType(enum.Enum):
    MISC = 0
    MOMENTUM = 1
    VOLUME = 2
    VOLATILITY = 3
    PRICETRANSFORM = 4
    CYCLE = 5
    PATTERN = 6
    STATISTICS = 7
    MATHTRANSFORM = 8
    MATHOPERATOR = 9
    OVERLAP = 10

@enum.global_enum
class IndicatorSource(enum.Enum):
    MISC = 0
    HARRY = 1
    TALIB = 2
    NUMBA = 3

class IndicatorDefinition(NamedTuple):
    name: str # short name of the indicator (e.g. 'SMA')
    long_name: str # long name (e.g. 'Simple Moving Average')
    type: IndicatorType
    src: IndicatorSource
    call_func: Callable
    parameters: Dict[str, Union[int, float, str]] = {}
    description: Optional[str] = None

it = IndicatorType
isrc = IndicatorSource
idef = IndicatorDefinition


def get_indicator_definitions() -> Dict[str, IndicatorDefinition]:
    pass

def get_indicator_definition(name: str) -> IndicatorDefinition:
    pass

# ---------------------------------------------------------------------------- #
#                                talib indicators                              #
# ---------------------------------------------------------------------------- #

# https://ta-lib.github.io/ta-doc/
#
#                                overlap studies                               #

# ==============================================================================
defs = {
'bbands' : IndicatorDefinition(
    name = 'BBANDS',
    long_name = 'Bollinger Bands',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.BBANDS,
    parameters = {
        'timeperiod': 20,
        'nbdevup': 2,
        'nbdevdn': 2,
        'matype': 0
    },
    description=''
),
'MA' : IndicatorDefinition(
    name = 'MA',
    long_name = 'Moving Average',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.MA,
    parameters = {
        'timeperiod': 10,
        'MAType': 0
    },
    description=''
),
'EMA' : IndicatorDefinition(
    name = 'EMA',
    long_name = 'Exponential Moving Average',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.EMA,
    parameters = {
        'timeperiod': 10,
        'EMAType': 0
    },
    description=''
),
'SMMA' : IndicatorDefinition(
    name = 'SMMA',
    long_name = 'Simple Moving Average',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.SMMA,
    parameters = {
        'timeperiod': 10,
        'MAType': 0
    },
    description=''
),
'WMA' : IndicatorDefinition(
    name = 'WMA',
    long_name = 'Weighted Moving Average',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.WMA,
    parameters = {
        'timeperiod': 10,
        'WMAType': 0
    },
    description=''
),
'VWMA' : IndicatorDefinition(
    name = 'VWMA',
    long_name = 'Volume-Weighted Moving Average',
    type = it.OVERLAP,
    src = isrc.TALIB,
    call_func=ta.VWMA,
    parameters = {
        'timeperiod': 10,
        'volumeperiod': 10
    },
    description=''
),
'ROC' : IndicatorDefinition(
    name = 'ROC',
    long_name = 'Rate of Change',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ROC,
    parameters = {
        'timeperiod': 10,
    },
    description=''
),
'RSI' : IndicatorDefinition(
    name = 'RSI',
    long_name = 'Relative Strength Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.RSI,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'Stochastic' : IndicatorDefinition(
    name = 'Stochastic',
    long_name = 'Stochastic Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.STOCH,
    parameters = {
        'fastk_period': 14,
        'slowk_period': 30,
        'slowk_matype': 0,
        'slowd_period': 30,
        'slowd_matype': 0
    },
    description=''
),
'STOCHRSI' : IndicatorDefinition(
    name = 'STOCHRSI',
    long_name = 'Stochastic Relative Strength Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.STOCHRSI,
    parameters = {
        'period': 14,
        'fastk_period': 14,
        'slowk_period': 30,
        'slowk_matype': 0,
        'slowd_period': 30,
        'slowd_matype': 0
    },
    description=''
),
'WILLR' : IndicatorDefinition(
    name = 'WILLR',
    long_name = 'Williams %R',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.WILLR,
    parameters = {
        'period': 14,
    },
    description=''
),
'ADX' : IndicatorDefinition(
    name = 'ADX',
    long_name = 'Average Directional Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ADX,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'ADXR' : IndicatorDefinition(
    name = 'ADXR',
    long_name = 'Average Directional Movement Index Rating',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ADXR,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'AROON' : IndicatorDefinition(
    name = 'AROON',
    long_name = 'Aroon Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.AROON,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'AROONOSC' : IndicatorDefinition(
    name = 'AROONOSC',
    long_name = 'Aroon Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.AROONOSC,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'CCI' : IndicatorDefinition(
    name = 'CCI',
    long_name = 'Commodity Channel Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.CCI,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'CMO' : IndicatorDefinition(
    name = 'CMO',
    long_name = 'Chande Momentum Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.CMO,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'ROCP' : IndicatorDefinition(
    name = 'ROCP',
    long_name = 'Rate of Change Percentage',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ROCP,
    parameters = {
        'timeperiod': 10,
    },
    description=''
),
'MFI' : IndicatorDefinition(
    name = 'MFI',
    long_name = 'Money Flow Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.MFI,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'TRIX' : IndicatorDefinition(
    name = 'TRIX',
    long_name = 'Triple Exponential Moving Average',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.TRIX,
    parameters = {
        'timeperiod': 10,
    },
    description=''
),
'ULTOSC' : IndicatorDefinition(
    name = 'ULTOSC',
    long_name = 'Ultimate Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ULTOSC,
    parameters = {
        'timeperiod1': 7,
        'timeperiod2': 14,
        'timeperiod3': 28,
    },
    description=''
),
'DX' : IndicatorDefinition(
    name = 'DX',
    long_name = 'Directional Movement Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.DX,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'PLUS_DI' : IndicatorDefinition(
    name = 'PLUS_DI',
    long_name = 'Positive Directional Indicator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.PLUS_DI,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'MINUS_DI' : IndicatorDefinition(
    name = 'MINUS_DI',
    long_name = 'Negative Directional Indicator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.MINUS_DI,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'DX_ADX' : IndicatorDefinition(
    name = 'DX_ADX',
    long_name = 'Directional Movement Index - ADX',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.DX_ADX,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'ADXR_ADX' : IndicatorDefinition(
    name = 'ADXR_ADX',
    long_name = 'Average Directional Movement Index Rating - ADX',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ADXR_ADX,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'APO' : IndicatorDefinition(
    name = 'APO',
    long_name = 'Absolute Price Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.APO,
    parameters = {
        'period': 14,
        'fastperiod': 3,
        'slowperiod': 10,
    },
    description=''
),
'MOM' : IndicatorDefinition(
    name = 'MOM',
    long_name = 'Momentum',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.MOM,
    parameters = {
        'timeperiod': 10,
    },
    description=''
),
'ROCMA' : IndicatorDefinition(
    name = 'ROCMA',
    long_name = 'Rate of Change - Moving Average',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ROCMA,
    parameters = {
        'timeperiod': 10,
        'matype': 0
    },
    description=''
),
'TSI' : IndicatorDefinition(
    name = 'TSI',
    long_name = 'True Strength Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.TSI,
    parameters = {
        'timeperiod': 14,
        'sign': 1
    },
    description=''
),
'WILLR_SLOW' : IndicatorDefinition(
    name = 'WILLR_SLOW',
    long_name = 'Williams %R - Slow',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.WILLR,
    parameters = {
        'period': 14,
        'slowperiod': 20
    },
    description=''
),
'ADOSC' : IndicatorDefinition(
    name = 'ADOSC',
    long_name = 'Accumulation/Distribution Oscillator',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.ADOSC,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'OBV' : IndicatorDefinition(
    name = 'OBV',
    long_name = 'On Balance Volume',
    type = it.VOLUME,
    src = isrc.TALIB,
    call_func=ta.OBV,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
'MFI_SLOW' : IndicatorDefinition(
    name = 'MFI_SLOW',
    long_name = 'Money Flow Index - Slow',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.MFI,
    parameters = {
        'timeperiod': 14,
        'slowperiod': 20
    },
    description=''
),
'FORCEINDEX' : IndicatorDefinition(
    name = 'FORCEINDEX',
    long_name = 'Force Index',
    type = it.MOMENTUM,
    src = isrc.TALIB,
    call_func=ta.FORCEINDEX,
    parameters = {
        'timeperiod': 14,
    },
    description=''
),
}

