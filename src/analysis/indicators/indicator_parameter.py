#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a class to define/manage parameters for indicators.



classes:
    Parameter
        parameter class

Created on Sat Aug 05 22:39:50 2023

@author: dhaneor
"""
import logging

from dataclasses import dataclass
from typing import (
    Literal,
    Union,
    Dict,
    Tuple,
    Optional,
    Iterable,
    Any
)
from talib import MA_Type


logger = logging.getLogger("main.parameter")
logger.setLevel(logging.INFO)

Params = Dict[str, Union[str, float, int, bool]]
IndicatorSource = Literal["talib", "nb"]

MA_TYPES = MA_Type.__dict__.get("_lookup", [])


@dataclass
class Parameter(Iterable):
    """Parameter class for indicators.

    A hardened class that encapsulates the functionality of
    parameters (and their parameter space) for technical indicators.
    It tries to prevent clients from setting parameters to values:

    - that do not make sense, ex: timeperiod=0
    - or would be rejected by the underlying  implementation of an
    indicator anyway. For instance, it is not possible to set negative
    timeperiods or negative values for 'matype'
    - that would change the type of the value (int to float is OK in
    some cases though)

    When changing the parameter space, some more checks are applied:

    - if the value is outside the hard limits for the parameter space
    (which were set during the instantiation of the indicator class using
    this class), an Exception is raised.
    - if the minimum is greater than the maximum, an exception is raised

    All of this is done to prevent sloppy humans (like me) from setting
    values that lead to problems further down the road. But even more so,
    this applies to optimizers that use these parameter values and the
    parameter space to optimize the performance of the indicator. They
    must not be allowed to mess up the system when running in production!
    """

    name: str
    initial_value: float | int | bool
    hard_min: float | int | bool
    hard_max: float | int | bool

    _value: float | int = None
    _enforce_int: bool = False

    step: Optional[float | int] = 1

    def __str__(self):
        return f"Parameter {self.name} -> {self.value}"

    def __iter__(self):
        return iter(range(self.hard_min, self.hard_max + 1, self.step))

    def __post_init__(self):
        self._value = self.initial_value

        for elem in ("period", "lookback", "type"):
            if elem in self.name.lower():
                self._enforce_int = True
                break

        self.hard_min = self._validate(self.hard_min)
        self.hard_max = self._validate(self.hard_max)

        if self.hard_min > self.hard_max:
            raise ValueError(
                f"hard_min ({self.hard_min}) > hard_max ({self.hard_max})"
                )

    @property
    def value(self) -> float | int | bool:
        """Returns the value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: float | int | bool) -> None:
        """Sets the value of the parameter.

        Raises
        ------
        TypeError
            if value not of the same type as the current value
        ValueError
            • if value is not in the allowed parameter space
            • if negative values for a timeperiod a requested
            • if negative values for categorical parameters are requested
        """
        self._value = self._validate(value)
        logger.info(f"Set parameter {self.name} to {self._value}")

    @property
    def space(self) -> Tuple[float | int | bool]:
        """Parameter space property. Setting not allowed after init."""
        return self.hard_min, self.hard_max, self.step

    @space.setter
    def space(self, space: Any) -> None:
        """Sets the parameter space.

        Raises
        ------
        PermissionError
            if an attempt is made to change the parameter space.
        """
        raise PermissionError("Changing the parameter space is not allowed.")

    def _validate(self, value):
        """Validates requested values for parameter"""
        # make sure the value we got has the correct type
        if not isinstance(value, type(self._value)):
            raise TypeError(
                f"invalid type for parameter {self.name}: {type(value)}"
                f" should be the same as {type(self._value)}"
            )

        if isinstance(self._value, str):
            return self._validate_string(value)

        if isinstance(self._value, (int, float)):
            return self._validate_numerical_value(value)

    def _validate_string(self, value):
        """Validates a string value

        Raises
        ------
        TypeError
            if value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(
                f"expected string for parameter {self.name}, but got: {type(value)}"
            )
        if len(value) > 50:
            logger.warning(
                "Value for parameter %s is too long (%s), cutting off the rest",
                self.name, value
                )
            value = value[:50]
        return value

    def _validate_numerical_value(self, value):
        # some values just can't be floats, flag is set in __post_init__
        value = int(round(value)) if self._enforce_int else value

        if not self.hard_min <= value <= self.hard_max:
            raise ValueError(
                f"parameter {self.name} out of range: " f"{value} not in {self.space}"
            )

        # timeperiods can't be 0 or negative
        if "period" in self.name and value == 0:
            logger.warning("%s == 0 does not make sense, setting to 1" % self.name)
            value = 1
        if "period" in self.name and value < 0:
            raise ValueError("%s cannot be negative, but was %s" % (self.name, value))

        return value

    def _get_steps(self):
        return range(self.hard_min, self.hard_max + 1, self.step)
