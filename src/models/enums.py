#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 02:16:23 2025

@author_ dhaneor
"""
from enum import Enum, unique

class OperandType(Enum):
    """Enums representing operator types."""

    INDICATOR = "indicator"
    SERIES = "series"
    TRIGGER = "trigger"
    VALUE_INT = "integer value"
    VALUE_FLOAT = "float value"
    BOOL = "boolean"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    

@unique
class COMPARISON(Enum):
    """Enums representing trigger conditions.
    define enums for different ways to compare two values during
    strategy execution. This makes sure that every strategy that
    is built by the strategy_builder() has a clearly defined
    comparison method, thereby reducing the possibility for errors
    when running the bot and using the strategy.
    """

    IS_ABOVE = "is above"
    IS_ABOVE_OR_EQUAL = "is above or equal"
    IS_BELOW = "is below"
    IS_BELOW_OR_EQUAL = "is below or equal"
    IS_EQUAL = "is equal"
    IS_NOT_EQUAL = "is not equal"
    CROSSED_ABOVE = "crossed above"
    CROSSED_BELOW = "crossed below"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name