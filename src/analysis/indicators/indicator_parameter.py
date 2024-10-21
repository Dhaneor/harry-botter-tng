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

from dataclasses import dataclass, field
from typing import (
    Literal,
    Sequence,
    Union,
    Dict,
    Optional,
    Iterable,
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
    this class), it is set to the closest valid value
    - if the minimum is greater than the maximum, an exception is raised

    All of this is done to prevent sloppy humans (like me) from setting
    values that lead to problems further down the road. But even more so,
    this applies to optimizers that use these parameter values and the
    parameter space to optimize the performance of the indicator. They
    must not be allowed to mess up the system when running in production!
    """

    name: str
    initial_value: float | int | bool
    hard_min: Optional[float | int | bool]
    hard_max: Optional[float | int | bool]

    _value: float | int | bool
    _enforce_int: bool = False
    _space: Sequence[float | int | bool | str] = field(default_factory=list)

    step: float | int = 1

    def __str__(self):
        return f"Parameter {self.name} -> {self.value}"

    def __iter__(self):
        return iter(range(self.hard_min, self.hard_max + 1, self.step))

    def __post_init__(self):
        for elem in ("period", "lookback", "type"):
            if elem in self.name.lower():
                self._enforce_int = True
                break

        if self.hard_min is not None:
            self.hard_min = self._validate(self.hard_min)
        if self.hard_max is not None:
            self.hard_max = self._validate(self.hard_max)

        self._space = self.hard_min, self.hard_max, self.step

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
            if value is not a float, int, bool or string.
        TypeError
            if type of value is incompatible with current value
        ValueError
            if value is not in the allowed parameter space.
        """
        if not isinstance(value, (float, int, bool, str)):
            raise TypeError(f"invalid type for parameter {self.name}: {type(value)}")

        if any(
            arg
            for arg in (
                (
                    isinstance(self.value, (float, int, bool))
                    & (not isinstance(value, (float, int, bool)))  # noqa: W503
                ),
                (isinstance(self.value, str) & (not isinstance(value, str))),
            )
        ):
            raise TypeError(
                "invalid type for parameter %s: %s" % (self.name, type(value))
            )

        if not self.space[0] <= value <= self.space[1]:
            raise ValueError(
                f"parameter {self.name} out of range: " f"{value} not in {self._space}"
            )

        self._value = self._validate(value)

    @property
    def space(self) -> Sequence[float | int | bool]:
        """Parameter space property."""
        if self.hard_min and self.hard_max:
            return self.hard_min, self.hard_max, self.step

        return self._space

    @space.setter
    def space(self, space: Sequence[float | int]) -> None:
        """Sets the parameter space.

        Raises
        ------
        TypeError
            if space is not a Sequence.
        ValueError
            if requested minimum > requested maximum
        ValueError
            if space does not contain 2 or 3 values.
        ValueError
            if space[0] < hard_min.
        ValueError
            if space[1] > hard_max.
        """
        # check that we have a sequence
        if not isinstance(space, Sequence):
            raise TypeError(
                f"invalid type for parameter space {self.name}: {type(space)}"
            )

        # check that we have 2 or 3 values
        if not (2 <= len(space) <= 3):
            raise ValueError(
                f"parameter space must contain 2 or 3 values, not {len(space)}"
            )

        # check that minimum < maximum
        if space[0] > space[1]:
            raise ValueError(f"parameter space minimum {space[0]} > maximum {space[1]}")

        # check against hard minimum and maximum
        if self.hard_min is not None and space[0] < self.hard_min:
            raise ValueError(
                f"parameter space minimum {space[0]} < hard minimum {self.hard_min}"
            )

        if self.hard_max is not None and space[1] > self.hard_max:
            raise ValueError(
                f"parameter space maximum {space[1]} > hard maximum {self.hard_max}"
            )

        self.hard_min = self._validate(space[0])
        self.hard_max = self._validate(space[1])
        self.step = self._validate(space[2]) if len(space) == 3 else 1

    def _validate(self, value):
        """Validates requested values for parameter.

        Raises
        ------
        TypeError
            if the type of the value does not match the expected type
        ValueError
            if negative values for a timeperiod a requested
        ValueError
            if negative values for categorical parameters are requested
        """
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

        # timeperiods can't be 0 or negative
        if "period" in self.name and value == 0:
            logger.warning("%s == 0 does not make sense, setting to 1" % self.name)
            value = 1
        if "period" in self.name and value < 0:
            raise ValueError("%s cannot be negative, but was %s" % (self.name, value))

        return value

    def _get_steps(self):
        return range(self.hard_min, self.hard_max + 1, self.step)
