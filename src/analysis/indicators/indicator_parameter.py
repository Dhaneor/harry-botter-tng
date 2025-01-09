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
import numpy as np

from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Dict,
    Tuple,
    Optional,
    Iterable,
    Sequence,
    Callable
)
from random import uniform, randint
from talib import MA_Type


logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.ERROR)
logger.debug("Initializing Parameter class")

Params = Dict[str, Union[str, float, int, bool]]
IndicatorSource = Literal["talib", "nb"]

MA_TYPES = MA_Type.__dict__.get("_lookup", [])


@dataclass
class Parameter(Iterable):
    """Parameter class for indicators.

    A hardened class that encapsulates the functionality of
    parameters (and their parameter space) for technical indicators.

    The class is iterable, which is used to iterate over the parameters
    during optimization (grid search) using the defined step size.

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
    step: Optional[float | int]
    on_change: Callable | None = None
    validate: bool = False
    _subscribers: set[Callable] = field(default_factory=set)
    _value: float | int = None
    _enforce_int: bool = False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.name}: {self.value}"

    def __iter__(self):
        return iter(
            np.arange(self.hard_min, self.hard_max, self.step)
            )

    def __post_init__(self):
        self._value = self.initial_value

        if self._value is None:
            raise ValueError(
                f"initial_value of parameter {self.name} is None"
                )

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

        if self.hard_min + self.step > self.hard_max:
            raise ValueError(
                f"'step' is too big: "
                f"hard_min ({self.hard_min}) + step ({self.step})"
                f" > hard_max ({self.hard_max})"
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
        self._value = self._validate(value) if self.validate else value

        if self._value is None:
            raise TypeError(f"Invalid value {self._value} for parameter {self.name}")

        # logger.info(f"Set parameter {self.name} to {self._value}")
        # logger.debug(f"Calling subscribers for parameter {self.name}")
        # for subscriber in self._subscribers:
        #     subscriber(self.name)

    @property
    def space(self) -> Tuple[float | int | bool]:
        """Parameter space property. Setting not allowed after init."""
        return self.hard_min, self.hard_max, self.step

    def increase(self, step: int | float | None = None) -> None:
        step = step or self.step
        self.value += step

    def decrease(self, step: int | float | None = None) -> None:
        step = step or self.step
        self.value -= step

    def randomize(self) -> None:
        """Randomizes the parameter value within the hard limits."""
        if self._enforce_int:
            value = randint(self.hard_min, self.hard_max)
        else:
            value = uniform(self.hard_min, self.hard_max)
            value = round(value / self.step) * self.step

        logger.debug("Randomizing parameter %s to %s", self.name, value)
        self.value = value

    def add_subscriber(self, callback: Callable) -> None:
        """Adds a callback function to the list of subscribers."""
        self._subscribers.add(callback)

    def as_dict(self) -> Params:
        """Returns a dictionary representation of the parameter."""
        return {self.name: self._value}

    # --------------------------------------------------------------------------------
    def _validate(self, value):
        """Validates requested values for parameter"""
        # make sure the value we got has the correct type
        self.validate_type(value)

        if isinstance(self._value, str):
            return self._validate_string(value)

        if isinstance(self._value, (int, float, np.int64)):
            return self._validate_numerical_value(value)

    def validate_type(self, value):
        """Validates a value based on its type.

        Raises
        ------
        TypeError
            if value is not of the correct type
        """
        if self._value is str:
            return self._validate_string(value)
        if isinstance(self._value, (int, float, np.int64)):
            return self._validate_numerical_value(value)

        raise TypeError(
            f"{self.name} / {value} -> expected {type(self._value)}"
            f", but got {type(value)}\n"
            f"{self.__dict__}"
            )

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
        logger.debug(f"Validating numerical value {value} for parameter {self.name}")

        if not isinstance(value, (int, float, np.number)):
            raise TypeError(
                f"expected numerical value for parameter {self.name}, "
                "but got: {type(value)}"
            )

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


if __name__ == "__main__":
    p = Parameter("timeperiod", 5, 1, 50, 1)
    print(p)
