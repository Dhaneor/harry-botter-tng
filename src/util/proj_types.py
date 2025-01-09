#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides all custom types used in this project.

Created on Fri August 11 21:44:23 2023

@author_ dhaneor
"""
from decimal import Decimal
from fractions import Fraction
from typing import Tuple, Any, Literal, TypeVar
import numpy as np
import numpy.typing as npt


NumberT = TypeVar('NumberT', float, Decimal, Fraction)

Numeric = np.flexible
Prices = Literal['open', 'high', 'low', 'close', 'volume']
Data = dict[str, npt.NDArray[Numeric]]
Weight = float  # type for all strategy weight values
Signals = dict[str, Tuple[np.ndarray, float]]
Parameters = dict[str, Any]
PlotParametersT = dict[str, Any]
OperandDefinitionT = tuple | Prices | dict[str, int | float | bool]

Array_1d = npt.NDArray[Numeric]
Array_2d = npt.NDArray[Numeric]
ArrayLikeT = TypeVar("ArrayLikeT", Array_1d, Array_2d)
