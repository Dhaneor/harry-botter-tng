#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a factory for single conditions.

Clients can use it to build a Condition object from a ConditionDefinition,

The factory function should be the only function from this module,
that is called by the user/client. It must be given a ConditionDefinition,
which is a formal description of the condition to be build.


classes:
    Comparison (enum)
        Enums for all comparison operators.

    ConditionDefinition
        Formal description of a trigger condition.

    Condition
        Produces a signal for one satisfied conditon. A Condition
        has two comparands and one operand two produce signals with
        its run() function.

    ConditionFactory
        A factory for building Condition objects from ConditionDefinitions.

functions:
    factory(c_def: ConditionDefinition) -> Condition
        simple wrapper so you don't have to instantiate the factory class

Example:
    >>> import condition as cnd
    >>>
    >>> # Create a ConditionDefinition
    >>> c_def = cnd.ConditionDefinition(
    >>>     left_operand=12,
    >>>     comparison=Comparison.EQUAL,
    >>>     right_operand=15
    >>> )
    >>>
    >>> # Generate a Condition using the factory function
    >>> condition = cnd.factory(c_def)
    >>>
    >>> # Trigger the condition and get the signal, data must be a
    >>> # dictionary with the following the keys
    >>> # "open", "close", "high", "low", "volume" and data as
    >>> # Numpy arrays.
    >>> result = condition.run(data)

Created on Sat Aug 18 11:14:50 2023

@author: dhaneor
"""

import itertools
import logging
from enum import Enum, unique
from dataclasses import dataclass
from numba import jit
from typing import Callable, NamedTuple, Iterable, Optional, TypeAlias
import numpy as np

from ..util import proj_types as tp
from ..util import comp_funcs as cmp
from . import operand as op

logger = logging.getLogger("main.condition")
logger.setLevel(logging.INFO)

# data for testing the correct functioning of a Condition class
TEST_DATA_SIZE = 500
TEST_DATA = {
    "open": np.random.random(TEST_DATA_SIZE),
    "close": np.random.random(TEST_DATA_SIZE),
    "high": np.random.random(TEST_DATA_SIZE),
    "low": np.random.random(TEST_DATA_SIZE),
    "volume": np.random.random(TEST_DATA_SIZE),
}


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

# Within the Condition class each arm or branch (for opening/closing
# long or short positions), is defined as a tuple with three elements:
# 1) a string for the comparison, e.g. CROSSED_ABOVE
# 2) another tuple that holds the names of the dictionary keys for the
#   values that should be compared against each other
# 3) the actual function object that is used to do the comparison,
#   which will be on eof the functions frim the cmp_funcs dictionary
#   defined above.
#
# These values are used in the execute method of the Condition class.
# They are set by the ConditionFactory - this explanation is just
# provided for better understanding of the .execute() method and the
# following TypeAlias:
ConditionBranch: TypeAlias = tuple[str, tuple[str, str], Callable]


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

    Attributes:
    ----------
    operands
        a list of all operands used by the condition
    comparand_a
        one output name of an Operand (in self.operands)
    comparand_b
        ...
    comparand_c
        ...
    open_long
        a tuple with a comparison function, and two operands
        that are the arguments for the comaprison function
    open_short
        ...
    close_short
        ...
    close_short
        ...
    """

    operand_a: Optional[op.Operand] = None
    operand_b: Optional[op.Operand] = None
    operand_c: Optional[op.Operand] = None
    operand_d: Optional[op.Operand] = None

    comparand_a: Optional[str] = None
    comparand_b: Optional[str] = None
    comparand_c: Optional[str] = None
    comparand_d: Optional[str] = None

    open_long: ConditionBranch | None = None
    open_short: ConditionBranch | None = None
    close_long: ConditionBranch | None = None
    close_short: ConditionBranch | None = None

    def __repr__(self) -> str:
        out = ["Condition("]
        for arm in ("open_long", "open_short", "close_long", "close_short"):
            if getattr(self, arm) is not None:
                a = getattr(self, arm)
                desc = f"{a[1][0]} --> {a[0]}  --> {a[1][1]}"
                out.append(f"\n\t{arm}:\t{desc}")
        out.append("\n)")
        return "".join(out)

    @property
    def operands(self) -> Iterable[op.Operand]:
        """Return all operators used in the condition"""
        return self.operand_a, self.operand_b, self.operand_c, self.operand_d

    @property
    def indicators(self) -> tuple[op.ind.Indicator]:
        """Return all indicators used in the condition

        This is for use by the optimizer(s).
        """
        return tuple(
            itertools.chain(
                self.operand_a.indicators
                if isinstance(self.operand_a, (op.OperandIndicator, op.OperandTrigger))
                else tuple(),
                self.operand_b.indicators
                if isinstance(self.operand_b, (op.OperandIndicator, op.OperandTrigger))
                else tuple(),
                self.operand_c.indicators
                if isinstance(self.operand_c, (op.OperandIndicator, op.OperandTrigger))
                else tuple(),
                self.operand_d.indicators
                if isinstance(self.operand_d, (op.OperandIndicator, op.OperandTrigger))
                else tuple(),
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
    def execute(self, data: tp.Data) -> ConditionResult:
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
        for operand in (self.operand_a, self.operand_b, self.operand_c, self.operand_d):
            if operand is not None:
                logger.debug("running operand %s", operand.name)
                operand.run(data)

        return ConditionResult(
            # open long
            open_long=self.open_long[2](
                data[self.open_long[1][0]], data[self.open_long[1][1]]
            )
            if self.open_long is not None
            else np.full_like(data["close"], False, dtype=bool),
            # open short
            open_short=self.open_short[2](
                data[self.open_short[1][0]], data[self.open_short[1][1]]
            )
            if self.open_short is not None
            else np.full_like(data["close"], False, dtype=bool),
            # close long
            close_long=self.close_long[2](
                data[self.close_long[1][0]], data[self.close_long[1][1]]
            )
            if self.close_long is not None
            else np.full_like(data["close"], False, dtype=bool),
            # close short
            close_short=self.close_short[2](
                data[self.close_short[1][0]], data[self.close_short[1][1]]
            )
            if self.close_short is not None
            else np.full_like(data["close"], False, dtype=bool),
        )

    def is_working(self) -> bool:
        """Checks if the condition is working with random data"""
        logger.debug("Checking if %s is working", self)

        test_res = self.execute(TEST_DATA)

        output_matches_size_input = all(
            arg
            for arg in (
                (
                    tr.shape == TEST_DATA["close"].shape
                    for tr in (
                        test_res.open_long,
                        test_res.open_short,
                        test_res.close_short,
                        test_res.close_long,
                    )
                )
            )
        )

        return all(
            arg
            for arg in (
                isinstance(test_res, ConditionResult),
                output_matches_size_input,
            )
        )


class ConditionFactory:
    def __init__(self):
        self.condition: Optional[Condition] = None

    def build_condition(self, condition_definition: ConditionDefinition) -> Condition:
        """Builds a Condition class from a ConditionDefinition class.

        Parameters
        ----------
        c_def: ConditionDefinition
            a condition definition class with all necessary parameters

        test_it: bool
            test whether the condition is working properly, defaults to False

        Returns
        -------
        Condition
            a condition class instance

        Raises
        ------
        ValueError
            if the requested comparison function is not supported
        KeyError
            if c_def is not a ConditionDefinition
        RuntimeError
            if the .run() method of the instance is not working correctly
        """
        self.condition = Condition()

        if not isinstance(condition_definition, ConditionDefinition):
            raise ValueError(f"{condition_definition} is not a ConditionDefinition")

        # start with building Operand instances, if they were defined
        for operand_name in ("operand_a", "operand_b", "operand_c", "operand_d"):
            operand_desc = getattr(condition_definition, operand_name, None)

            if getattr(condition_definition, operand_name, None) is not None:
                setattr(self.condition, operand_name, op.operand_factory(operand_desc))

                logger.debug("...built operand %s", op.operand_factory(operand_desc))

        # now let's check the sub-conditions for opening and closing
        # longs and/or shorts
        for arm in ("open_long", "open_short", "close_long", "close_short"):
            if (arm_def := getattr(condition_definition, arm, None)) is None:
                continue

            match arm_def:
                case str():
                    setattr(self.condition, arm, self._from_string(arm_def))
                case tuple():
                    setattr(self.condition, arm, self._from_tuple(arm_def))
                case _:
                    raise ValueError(f"{arm_def} is not supported")

        return self.condition

    def _from_string(self, desc: str):
        logger.debug("...building arm from string: %s", desc)
        splitted = desc.split(" ")
        comparison = COMPARISON(" ".join(splitted[1:-1]))
        return self._from_tuple((splitted[0], comparison, splitted[-1]))

    def _from_tuple(self, desc: tuple):
        """Build an Operand instance from a tuple (description)"""

        left_arm, right_arm = desc[0], desc[-1]
        comparison, comparands = desc[1], [None, None]

        # process left and right arm of the sub-condition
        for idx, op_def in enumerate((left_arm, right_arm)):
            output = None

            # handle cases where specific outpüut is requested
            if isinstance(op_def, str) and "." in op_def:
                op_def, output = op_def.split(".")

            # each arm of this sub_condition can be either the name of an
            # indicator or a reference to an operand (= "a" .. "c") that
            # was defined in the condition definition.
            if op_def in ("a", "b", "c", "d"):
                if (operand := getattr(self.condition, f"operand_{op_def}")) is None:
                    raise ValueError(f"no operand_{op_def} defined for {desc}")
            else:
                if isinstance(op_def, (int, float, bool)):
                    op_def = self._get_fixed_indicator_definition(op_def)

                operand = op.operand_factory(op_def)
                self._set_operand(operand)

            comparands[idx] = self._get_output_name(operand, output)

        [self._set_comparand(c) for c in comparands]

        return comparison, comparands, self._prep_comp_func(comparison)

    def _get_output_name(self, operand: op.Operand, output: str | None) -> str:
        logger.debug("...getting output name for: %s", output)
        if output is None:
            return operand.output
        else:
            for out_name in operand.output_names:
                if output in out_name:
                    return out_name
            else:
                raise ValueError(f"no output named {output} found in {operand}")

    def _set_operand(self, operand: op.Operand) -> None:
        logger.debug("...setting operand for: %s", operand.output)
        for operand_name in ("operand_a", "operand_b", "operand_c", "operand_d"):
            if getattr(self.condition, operand_name) == operand:
                break

            if getattr(self.condition, operand_name) is None:
                setattr(self.condition, operand_name, operand)
                break

    def _set_comparand(self, output: str) -> None:
        logger.debug("...setting comparand for: %s", output)
        for comparand_name in (
            "comparand_a",
            "comparand_b",
            "comparand_c",
            "comparand_d",
        ):
            if getattr(self.condition, comparand_name) == output:
                break

            if getattr(self.condition, comparand_name) is None:
                setattr(self.condition, comparand_name, output)
                break

    def _get_fixed_indicator_definition(self, value: tp.Numeric) -> tuple:
        return ("trigger", value)

    def _prep_comp_func(self, comp: COMPARISON) -> Callable:
        """Selects a comparison function based on a condition definition.

        Parameters
        ----------
        comparison
            the requested comparison, ex: COMPARISON.IS_ABOVE

        Returns
        -------
        Callable
            a comparison function based on the given condition definition
        """
        try:
            return cmp_funcs[comp]
        except KeyError as err:
            logger.error("invalid comparison function: %s", comp)
            raise ValueError(f"invalid comparison function: {comp}") from err


# ======================================================================================
condition_factory = ConditionFactory()


def factory(c_def: ConditionDefinition, test_it: bool = False) -> Condition:
    """Builds a Condition class from a ConditionDefinition class.

    This function prepares the (executable) Condition class(es) for a
    strategy. This happens here, during building of the strategy class:
    a) to make sure they work properly before using them
    b) to save time when the conditions are executed

    If test_it is True, the run() method of the condition will be
    tested with synthetic data before giving it to the caller. This
    is for convenience and to make sure that we are not using faulty
    instances of Condition (for instance, maybe the  ConditionDefiniton
    was written poorly and the operator.factory() did not catch the
    error).

    Parameters
    ----------
    c_def: ConditionDefinition
        a condition definition class with all necessary parameters

    test_it: bool
        test whether the condition is working properly, defaults to False

    Returns
    -------
    Condition
        the condition class

        This condition class has three elements:
        - comparand_a: a ready-to-go the indicator class or a
        numeric/boolean value, or the name of a price series
        (e.g 'close').
        - comparand_b: same as a
        - cmp_func: a comparison function to compare a & b

        Everything is ready for use in the signal generater of the
        strategy class.

    Raises
    ------
    ValueError
        if the requested comparison function is not supported
    KeyError
        if the value for c_def is not a ConditionDefinition
    RuntimeError
        if the .run() method of the instance is not working correctly
    """
    condition = condition_factory.build_condition(c_def)

    if test_it and not condition.is_working():
        raise RuntimeError(f"condition {condition} is not working")

    return condition
