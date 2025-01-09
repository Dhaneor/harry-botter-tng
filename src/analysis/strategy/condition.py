#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a factory for single conditions.

Clients can use it to build a Condition object from a ConditionDefinition,

classes:
    Comparison (enum)
        Enums for all comparison operators.

    ConditionDefinition
        Formal description of a trigger condition.

    ConditionResult
        A class to store the result of a condition evaluation in a 
        standardized way. Instances can be added, or compared with 
        AND/OR.

    Condition
        Produces a signal for one satisfied conditon. A Condition
        has two comparands and one operand two produce signals with
        its run() function.

Created on Sat Aug 18 11:14:50 2023

@author: dhaneor
"""

import logging
from enum import Enum, unique
from dataclasses import dataclass, field
from itertools import chain
from numba import jit
from typing import Callable, NamedTuple, Optional, Sequence
import numpy as np

from util import proj_types as tp
from ..util import comp_funcs as cmp
from . import operand as op

logger = logging.getLogger("main.condition")
logger.setLevel(logging.ERROR)


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

ConditionT =  dict[tuple[str, COMPARISON, str]]
AndConditionsT = Sequence[ConditionT]
OrConditionsT = Sequence[AndConditionsT]
ConditionDefinitionT = dict[tuple[str, COMPARISON, str]]


cmp_funcs = {
    COMPARISON.IS_ABOVE: cmp.is_above,
    COMPARISON.IS_ABOVE_OR_EQUAL: cmp.is_above_or_equal,
    COMPARISON.IS_BELOW: cmp.is_below,
    COMPARISON.IS_BELOW_OR_EQUAL: cmp.is_below_or_equal,
    COMPARISON.IS_EQUAL: cmp.is_equal,
    COMPARISON.IS_NOT_EQUAL: cmp.is_not_equal,
    COMPARISON.CROSSED_ABOVE: cmp.crossed_above,
    COMPARISON.CROSSED_BELOW: cmp.crossed_below,
}


@jit(nopython=True)
def merge_signals_nb(open_long, open_short, close_long, close_short):
    """Merges the four possible signals into one column."""
    n = len(open_long)
    signal = np.zeros(n, dtype=np.float64)
    position = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            if open_long[i] > 0:
                signal[i] = open_long[i]
                position[i] = 1
            elif open_short[i] > 0:
                signal[i] = open_short[i] * -1
                position[i] = -1
        else:
            prev_position = position[i - 1]
            signal[i] = signal[i - 1]
            position[i] = prev_position

            if open_long[i] > 0:
                signal[i] = open_long[i]
                position[i] = 1

            elif close_long[i] > 0:
                if prev_position > 0:
                    signal[i] = 0
                    position[i] = 0

            elif open_short[i] > 0:
                signal[i] = open_short[i] * -1
                position[i] = -1

            elif close_short[i] > 0:
                if prev_position < 0:
                    signal[i] = 0
                    position[i] = 0

    return signal, position


# ======================================================================================
class ConditionDefinition(NamedTuple):
    """A single condition for producing a signal.

    A condition is a function that takes two values and returns a
    boolean. This is only a definition and sending it to the
    build_condition() function will return the actual condition.
    One or more conditions can be combined to produce a
    signal.

    val_a and val_b can each be a tuple, a string or a fixed number.

    If a (3-element) tuple is given, it will be interpreted as:

        - index 0: the name of an indicator (or other pre-defined
        indicator-like functions (e.g. 'sentiment') that returns
        a 1D or 2D Numpy array. The most basic form for a single
        indicator with default inputs and default parameters
        would be:

        >>> ('sma',)

        - index 1..n: the data (=input - time series, array) to be
        passed to the indicator .

        Only 'open', 'high', 'low', 'close' and 'volume', are supported
        for this, But, his value can also be another tuple, describing
        another indicator. That makes it possible to describe for
        instance the ATR for the RSI, or the SMA over a base indicator.
        Common sense is the only limiting factor for the number of these
        nested indicators. Most indicators take only one input (usually
        close prices), but some, like the ATR, require more. If unsure,
        check the indicator with its help() method for required inputs!
        The ATR for instance would be defined like this:

        >>>    ('atr', 'open', 'high', 'low' {'timeperiod': 14})

        But in most cases, you can just omit this parameter, the default
        values as requested by the indicator will be used then.

        >>>   ('rsi', {'timeperiod': 14})

        - index -1: the parameters that will be passed to the indicator

        >>>    ('sma', 'close', {'timeperiod': 14})


    If a string is given, then the interpretation depends on the
    concrete value. For 'open', 'high', 'low', 'close'
    and 'volume' it will be interpreted as the name of a
    price series in the OHLCV data dictionary (that is passed to the
    Condition.execute() method). This can also be the name of an
    indicator ('sma', 'macd', etc) and is equiivalent to a tuple
    definition with no parameters: 'sma' == ('sma',)

    If a fixed number is given,it will be interpreted as the fixed
    value to compare against.

    By formalizing these definitions, it is possible to build a
    lot of different strategies dynamically, without having to write
    a dedicated class for each one. It also helps with running
    multiple backtests, iterating through parameter combinations,
    and building the strategy classes in a standardized way.

    TODO: This class could also be build from a string, for instance:
    '(rsi, close, {'timeperiod': 14}) is above 70', which may
    or may not make sense. :D


    Attributes:
    ----------
    symbol: str | None
        the symbol to use for the condition

    interval
        the interval ('1m', '1h', ... '1w') to use for the condition

    fetch_data_fn: Callable | None
        a function to fetch the data for the symbol and interval,

    comparand_a
        first value to compare

    comparand_b
        second value to compare, optional

    comparand_c
        third value to compare, optional

    comparand_d
        fourth value to compare, optional

    trigger
        the method used to compare the values (trigger condition),
        possible values are:

        IS_ABOVE, IS_ABOVE_OR_EQUAL, IS_BELOW, IS_BELOW_OR_EQUAL,
        IS_EQUAL, IS_NOT_EQUAL, CROSSED_ABOVE, CROSSED_BELOW, CROSSED

        will be used like in this example:
        bool(val_a > val_b), for trigger: IS_ABOVE

    """
    symbol: str | None = None
    interval: str | None = None
    fetch_data_fn: Callable | None = None

    operand_a: Optional[tp.OperandDefinitionT] = None
    operand_b: Optional[tp.OperandDefinitionT] = None
    operand_c: Optional[tp.OperandDefinitionT] = None
    operand_d: Optional[tp.OperandDefinitionT] = None

    open_long: Optional[str] = None
    open_short: Optional[str] = None
    close_long: Optional[str] = None
    close_short: Optional[str] = None


@dataclass
class ConditionResult:
    """
    Class that represents the result of one or more conditions.

    It is possible to combine multiple ConditionResult instances by:
    • Addition
    • Logical AND
    • Logical OR

    Raises:
    -------
    ValueError
        If initialized empty.
    """

    open_long: np.ndarray | None = None
    open_short: np.ndarray | None = None
    close_long: np.ndarray | None = None
    close_short: np.ndarray | None = None

    _combined_signal: np.ndarray | None = None

    def __post_init__(self):
        all_actions = (
            self.open_long,
            self.open_short,
            self.close_long,
            self.close_short,
        )

        not_none = next(filter(lambda x: x is not None, all_actions), None)

        if not_none is None:
            raise ValueError(
                "ConditionResult is empty - "
                "at least one action needs to be an array."
            )

        for action in ("open_long", "open_short", "close_long", "close_short"):
            if (elem := getattr(self, action)) is None:
                elem = np.full_like(not_none, fill_value=0, dtype=np.float64)
            setattr(self, action, elem.astype(np.float64))

    def __len__(self):
        return len(self.open_long)

    def __add__(self, other) -> "ConditionResult":
        res = []

        for attr in ["open_long", "open_short", "close_long", "close_short"]:
            res.append(np.add(getattr(self, attr), getattr(other, attr)))

        return ConditionResult(
            open_long=res[0],
            open_short=res[1],
            close_long=res[2],
            close_short=res[3],
        )

    def __and__(self, other) -> "ConditionResult":
        res = []

        for attr in ["open_long", "open_short", "close_long", "close_short"]:
            if (getattr(self, attr) is None) | (getattr(other, attr) is None):
                res.append(None)
            else:
                res.append(np.logical_and(getattr(self, attr), getattr(other, attr)))

        return ConditionResult(
            open_long=res[0],
            open_short=res[1],
            close_long=res[2],
            close_short=res[3],
        )

    def __or__(self, other) -> "ConditionResult":
        res = []

        for attr in ["open_long", "open_short", "close_long", "close_short"]:
            if (getattr(self, attr) is None) | (getattr(other, attr) is None):
                res.append(None)
            else:
                res.append(np.logical_or(getattr(self, attr), getattr(other, attr)))

        return ConditionResult(
            open_long=res[0],
            open_short=res[1],
            close_long=res[2],
            close_short=res[3],
        )

    def __ffill_array(self, arr):
        for idx in range(1, len(arr)):
            if arr[idx] == 0:
                arr[idx] = arr[idx - 1]
        return arr

    def ffill(self):
        for action in ("open_long", "open_short", "close_long", "close_short"):
            arr = self.__ffill_array(getattr(self, action))
            setattr(self, action, arr)

        return self

    def apply_weight(self, weight: float) -> "ConditionResult":
        return ConditionResult(
            open_long=np.multiply(self.open_long, weight),
            open_short=np.multiply(self.open_short, weight),
            close_long=np.multiply(self.close_long, weight),
            close_short=np.multiply(self.close_short, weight),
        )

    def as_dict(self):
        return {
            "open_long": self.open_long,
            "open_short": self.open_short,
            "close_long": self.close_long,
            "close_short": self.close_short,
            "signals": self.combined_signal,
        }

    @property
    def combined_signal(self):
        signal, _ = merge_signals_nb(
            self.open_long, self.open_short, self.close_long, self.close_short
        )

        return np.nan_to_num(signal)

    @combined_signal.setter
    def combined_signal(self, value):
        self._combined_signal = value

    @classmethod
    def from_combined(cls, combined: np.ndarray):
        """Method to build a ConditionResult object from an array.

        This helps to reverse the result from the .combined() method
        (see above) which produces a single column/array with the
        signals from a ConditionResult object.
        """
        open_long = np.zeros_like(combined, dtype=np.float64)
        close_long = np.zeros_like(combined, dtype=np.float64)
        open_short = np.zeros_like(combined, dtype=np.float64)
        close_short = np.zeros_like(combined, dtype=np.float64)

        position = 0

        for i in range(combined.shape[0]):
            if combined[i] > 0:
                open_long[i] = combined[i]
                position = 1
            elif combined[i] < 0:
                open_short[i] = abs(combined[i])
                position = -1
            else:
                if position == 1:
                    close_long[i] = 1
                elif position == -1:
                    close_short[i] = 1
                position = 0

        cr = ConditionResult(open_long, open_short, close_long, close_short)
        cr.combined_signal = combined

        return cr

    def _fill_nan_with_last(self, arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]

    def _add_arrays_with_nan_handling(self, arr1, arr2):
        # Fill NaN values in both arrays
        filled_arr1 = self._fill_nan_with_last(arr1)
        filled_arr2 = self._fill_nan_with_last(arr2)

        # Add the filled arrays
        result = filled_arr1 + filled_arr2

        return result


@dataclass
class Condition:
    """A condition, prepared for execution.

    The conditions are defined in the SignalGeneratorDefinition, which
    contains a dictionary of the following format:

    :: code ::
        conditions=dict(
            open_long=[
                ("aroon", COMPARISON.CROSSED_ABOVE, "aroon_trigger"),
                ("efficiency_ratio", COMPARISON.IS_ABOVE, "er_trending"),
            ],
            close_long=[
                ("aroon", COMPARISON.CROSSED_BELOW, "aroon_trigger"),
                ("efficiency_ratio", COMPARISON.IS_BELOW, "er_trending"),
            ],
            open_short=None,
            close_short=None,
        )

    The tuple in each list contains a the name of two operands, and a
    comparison operator. The operands are defined in another dictionary,
    and the operand names defined for the condition must correspond to
    a key in the operands dictionary (where the value is a working
    instance of an operand - an indicator, a fixed value, or a price 
    series).

    Attributes:
    ----------
    open_long
        a tuple with a comparison function, and two operands
        that are the arguments for the comaprison function
    open_short
        ...
    close_short
        ...
    close_short
        ...
    operands: dict[int, str]
        A dictionary, mapping operand names to their working instances

    """

    open_long: tuple[str, COMPARISON, str] | None = None
    open_short: tuple[str, COMPARISON, str] | None = None
    close_long: tuple[str, COMPARISON, str] | None = None
    close_short: tuple[str, COMPARISON, str] | None = None

    operands: dict[str, op.Operand] = field(default_factory={})

    def __repr__(self) -> str:
        return (
            f"[Conditions]\n"
            f"  open_long: {self.open_long}\n  close_long: {self.close_long}\n"
            f"  open_short: {self.open_short}\n  close_short: {self.close_short}"
        )

    @property
    def indicators(self) -> tuple[op.ind.Indicator]:
        """Return all indicators used in the condition

        This is for use by the optimizer(s).
        """
        return tuple(
            chain.from_iterable(
                op.indicators for op in self.operands.values()
            )
        )

    @property
    def plot_desc(self) -> tp.PlotParametersT:
        """Returns all necessary plot parameters for the condition.

        Returns
        -------
        tp.PlotParams
            plot parameters for the condition
        """
        return sum(
            filter(
                lambda x: x is not None,
                (
                    self.operand_a.plot_desc if self.operand_a else None,
                    self.operand_b.plot_desc if self.operand_b else None,
                    self.operand_c.plot_desc if self.operand_c else None,
                    self.operand_d.plot_desc if self.operand_d else None,
                ),
            )
        )

    # ..................................................................................
    def execute(self, data: tp.Data) -> dict:
        """Execute the condition.

        Parameters
        ----------
        data: tp.Data
            the OHLCV data dictionary

        Returns:
        --------
        ConditionResult
            A condition result object with four attached arrays for each of:
            open_long, open_short, close_long, close_short
        """
        return dict(
            open_long=self._execute_type("open_long", data),
            open_short=self._execute_type("open_short", data),
            close_long=self._execute_type("close_long", data),
            close_short=self._execute_type("close_short", data),
        )

    # ............................... helper methods ..................................
    def _execute_type(self, type_str: str, data: dict[str, np.ndarray]) -> np.ndarray:
        conditions = getattr(self, type_str)

        if conditions is None:
            return np.full(data["close"].shape, np.nan)

        result = None

        for condition in conditions:
            left = self.operands.get(condition[0])
            right = self.operands.get(condition[2])
            comp_fn = cmp_funcs.get(condition[1])

            logger.debug("left: %s, right: %s", left, right)

            single_condition_result = comp_fn(left.run(), right.run())

            if result is None:
                result = single_condition_result
            else:
                result = np.logical_and(result, single_condition_result)

        return result


class ConditionParser:
    """Factory for parsing/transforming condition definitions."""

    def __init__(self, operands: dict[str, op.Operand]) -> None:
        self.operands = operands

    def parse(self, conditions: ConditionDefinitionT) -> ConditionDefinitionT:
        # The conditions definitions in the dictionary are either in the 
        # form of list[tuple, ...] or in the form of list[list[tuple,...]].
        # We need to convert the former to the latter.
        for k, v in conditions.items():
            if v is not None:
                if not isinstance(v[0][0], list | tuple):
                    conditions[k] = [v]

        # ... now we can process them
        for k,v in conditions.items():
            if v is not None:
                conditions[k] = self._parse_or_conditions(v)

        return conditions

    def _parse_or_conditions(self, or_conditions: OrConditionsT) -> OrConditionsT:
        return [self._parse_and_conditions(elem) for elem in or_conditions] 

    def _parse_and_conditions(self, and_conditions: AndConditionsT) -> AndConditionsT:
        return [self._parse_condition(elem) for elem in and_conditions]

    def _parse_condition(self, condition: ConditionT) -> ConditionT:
        left, operator, right = condition

        for operand_name in (left, right):
            if operand_name not in self.operands:
                raise ValueError(
                    f"Operand '{operand_name}' not found in operands. "
                    f"Available operands: {', '.join(list(self.operands.keys()))}"
                    )
   
        if isinstance(operator, str):
            if operator.upper() not in COMPARISON:
                raise ValueError(operator)
            operator = COMPARISON[operator]
        elif not isinstance(operator, COMPARISON):
            raise ValueError(f"Invalid operator: {operator}")
        
        return left, operator, right

                