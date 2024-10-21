#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides Operand classes and their factory function.

TODO:   move the functionality from FixedIndicator back into OperandTrigger?!
        pro:
            - no need to have two classes for the same thing
            - simpler code
            - less complexity
        con:
            - different operand types have different level of abstraction
            - optimizer class will probably get more complicated. but will it really?

classes:
    PriceSeries
        enums for different price series

    OperandType
        enums for different operand types

    Operand
        base class for all operands

    OperandIndicator
        class for indicator operands

    OperandTrigger
        class for all operands that represent a fixed trigger value

    OperandSeries
        class for all operands that represent a price series

functions:
    operand_factory
        Factory function for creating operand instances from
        formalized operand descriptions. This function is used by
        the factory function for Condition classes.

Created on Sat Aug 18 10:356:50 2023

@author: dhaneor
"""
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from numbers import Number
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..indicators import indicator as ind
from ..util import proj_types as tp

logger = logging.getLogger("main.operand")
logger.setLevel(logging.ERROR)

# build a list of all available indicators, that can later be used
# to do a fast check if requested indicators are available.
ALL_INDICATORS = set(i.lower() for i in ind.get_all_indicator_names())

OperandDefinitionT = tuple | str


# ======================================================================================
@unique
class PriceSeries(Enum):
    """Enums representing price inputs."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def __contains__(self, item) -> bool:
        return next((True for member in self if member.value == item), False)


class OperandType(Enum):
    """Enums representing operator types."""

    INDICATOR = "indicator"
    SERIES = "series"
    VALUE_INT = "integer value"
    VALUE_FLOAT = "float value"
    BOOL = "boolean"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


VALID_PRICE_INPUTS = {member.value for member in PriceSeries}


# ======================================================================================
@dataclass
class Operand(ABC):
    name: str
    type_: OperandType
    inputs: tuple = field(default_factory=tuple)
    unique_name: str = ""
    output: str = ""

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    @abstractmethod
    def plot_desc(self) -> dict[str, tp.Parameters]:
        ...

    @abstractmethod
    def run(self, data: tp.Data) -> str:
        ...

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        ...


@dataclass(kw_only=True)
class OperandIndicator(Operand):
    """A single operand of a condition.

    A single operand can represent either:
        - an indicator
        - a series
        - a numerical value or a boolean.

    Attributes
    ----------
    name : str
        name of the operand

    type_ : OperandType
        the type of the operand.

    run_func : Optional[Callable]
        A function to run when calling the 'run' method of this operand.

    inputs : tuple
        the inputs for the indicator, for instance 'close', this
        field only applies to indicators

    unique_name : str
        the unique name of the operand, for instance 'sma_30'

    output_names: tuple[str, ...]
        The key/column names that are added to the data dict when
        running the operand.

    output : str
         The relevant output for the indicator, for instance
         'macd_signal'. This is the same as output_names for
         most indicators, except the  ones that return multiple
         values.

    parameters : TypedDict
        parameters for the indicator, only applies to indicators

    parameter_space : Optional[Sequence[Number]]
        A range of values to use for automated optimizing of the
        parameters.
    """

    name: str
    type_: OperandType
    indicator: ind.Indicator | None = field(default=None)
    run_func: Callable | None = field(default=None)
    inputs: tuple = field(default_factory=tuple)
    unique_name: str = ""
    output_names: tuple = field(default_factory=tuple)
    output: str = ""
    parameters: dict = field(default_factory=dict)
    parameter_space: Optional[Sequence[Number]] = None
    indicators: list[ind.Indicator] = field(default_factory=list)

    def __repr__(self) -> str:
        name = f"{self.name.upper()}" if isinstance(self.name, str) else f"{self.name}"
        type_ = f"[{self.type_}] "

        if not self.indicator:
            return f"{type_}{name}"

        match self.type_:
            case OperandType.INDICATOR:
                in_str = f" of {self.inputs}" if self.inputs else ""
                param_str = f" with {self.indicator.parameters}"
            case OperandType.SERIES:
                in_str = f" of {self.inputs}" if self.inputs else ""
                param_str = ""
            case OperandType.VALUE_INT | OperandType.VALUE_FLOAT | OperandType.BOOL:
                in_str = ""
                param_str = ""
            case _:
                logger.error("Invalid operand type: %s", self.type_)

        return f"{type_} {name}{param_str}{in_str}"

    def __post_init__(self) -> None:
        # make sure, inputs is a tuple or None
        if self.inputs is not None:
            self.inputs = (
                tuple((self.inputs))
                if not isinstance(self.inputs, tuple)
                else self.inputs
            )

        if not self.indicator:
            logger.error("Operand %s has no indicator", self.name)
            raise AttributeError(f"Operand {self.name} has no indicator")

        self.unique_name = self.indicator.unique_name
        self.output_names = tuple(self.indicator.unique_output)

        self._update_names(self.unique_name, self.indicators)
        self.parameters = {self.unique_name: i.parameters for i in self.indicators}

    @property
    def display_name(self) -> str:
        """Return the display name for the operand"""
        params = " ".join((str(x) for x in self.indicator.parameters.values()))
        if len(self.indicators) == 1:
            return f"{self.indicator.display_name} ({params})"
        else:
            out = [f"{self.indicator.display_name} ({params}) of "]

            for indicator in self.indicators[1:]:
                params = " ".join((str(x) for x in indicator.parameters.values()))
                out.append(f"{indicator.name.upper()} ({params})")
            return " ".join(out)

    @property
    def plot_desc(self) -> dict[str, tp.Parameters]:
        """Return the plot description for the operand.

        Takes the PlotDescription as returned by all indicators for
        the operand and returns a PlotDescription for the operand,
        after combining them into one. Results depend on the type of
        the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """
        desc = self.indicator.plot_desc

        logger.debug(desc.lines)
        logger.debug(self.output_names)

        lines = [
            (name, line[1])
            for name in self.output_names
            for line in desc.lines
            if name.startswith(f"{line[0]}_")
        ]

        # put the names of all outputs that need to be plotted as
        # a channel (for instance Bollinger Bands) into the list
        # for the channel prpoerty of the PlotDescription
        channel = (
            ()
            if not desc.channel
            else (
                elem
                for elem in self.output_names
                if ("upper" in elem) | ("lower" in elem)
            )
        )

        hist = (
            ()
            if not desc.hist
            else (elem for elem in self.output_names if ("hist" in elem))
        )

        return ind.PlotDescription(
            label=self.display_name,
            is_subplot=any(arg.is_subplot for arg in self.indicators),
            lines=lines,
            triggers=desc.triggers,
            channel=list(channel),
            hist=list(hist),
            level="operand",
        )

    # ..........................................................................
    def update_parameters(self, params: dict[str, tp.Parameters]) -> None:
        """Update the parameters for the indicator(s).

        Parameters
        ----------
        params : dict[str, tp.Parameters]
            The new parameters, keys are the names of the indicators.
            The keys/names in this dict will be checked against the
            unique names (e.g. sma_30) of all indicators that this
            operand has and/or uses as inputs. If these match, the
            indicator will be updated with the new parameters (= the
            dictionary that is the value for this key). Keys that do
            not match any unique name will be ignored.

            .. code-block:: python
            {
                'sma_30': {'timeperiod': 28},
                'rsi_14': {'timeperiod': 7}
            }

        Note: This is meant to be used by optimmizers. If you want to
        use a strategy with different parameters, it may be easier to
        create a new strategy instance with the new parameters.
        """

        logger.info("parameter update request %s for %s", params, self.unique_name)

        post_init = False

        for k in params.keys():
            logger.info("processing update request for : %s", k)

            for indicator in self.indicators:
                if k == indicator.unique_name:
                    logger.info(
                        "...updating parameter %s for %s with %s",
                        k,
                        indicator.unique_name,
                        params[k],
                    )
                    indicator.parameters = params[k]
                    post_init = True
                else:
                    logger.debug("... %s not relevant for %s", k, indicator.unique_name)

        if post_init:
            self.__post_init__()

    def run(self, data: tp.Data) -> str:
        """Run the operand on the given data.

        Parameters
        ----------
        data : tp.Data
            OHLCV dictionary

        Returns
        -------
        np.ndarray
            resulting array from running the operand

        Raises
        ------
        ValueError
            if the run function is not defined for the operand
        """
        return self._run_indicator(data)

    def as_dict(self):
        """Return a dictionary representation of the operand.

        Returns
        -------
        dict[str, Any]
            a dictionary representation of the operand
        """
        return asdict(self)

    # ..........................................................................
    def _run_indicator(
        self,
        data: tp.Data,
        indicator_: ind.Indicator | None = None,
        level: int = 0,
    ) -> str:
        """Runs the indicator.

        Parameters
        ----------
        data : tp.Data
            OHLCV dictionary
        indicator_ : ind.Indicator | None, optional
            indicator instance, default: None
        level : int, optional
            recursion level for nested indicators, by default 0

        Returns
        -------
        str
            keys in the data dictionary that were added by this indicator

        Raises
        ------
        ValueError
            if unable to get indicator instance
        """
        indicator_ = indicator_ or self.indicator

        if indicator_ is None:
            raise ValueError(f"No indicator defined for {self.name}")

        # when using the same indicator in different conditions,
        # there's no need to run it multiple times
        if indicator_.unique_name in data:
            logger.info(
                "[%s] Indicator %s already in data, skipping",
                level,
                indicator_.name,
            )
            return self.output_names if level > 0 else self.output_names[0]

        logger.info("[%s] Running indicator: %s", level, indicator_.name)

        indicator_values = indicator_.run(
            *self._get_ind_inputs(self.inputs, data, len(indicator_.input), level)
        )

        # the indicator may return a single array, or a tuple of arrays
        if isinstance(indicator_values, np.ndarray):
            data[self.output_names[0]] = indicator_values

        elif isinstance(indicator_values, tuple):
            data.update(
                {
                    key: indicator_values[idx]
                    for idx, key in enumerate(self.output_names)
                }
            )

        else:
            logger.error("Indicator did not return an array or tuple of arrays")
            raise ValueError(
                f"Indicator {indicator_.name} did not return an array/tuple of arrays"
            )

        # return name of relevant key that was added to the data dict
        return self.output_names if level > 0 else self.output

    def _get_ind_inputs(
        self, req_in: str | tuple, data: tp.Data, max_inputs: int, level: int
    ) -> tuple[Any, ...]:
        """Get the required inputs for an indicator.

        NOTE: The method can make recursive calls if one or (unlikely)
        more of the requested inputs describe(s) another indicator.
        Theoretically it is possible to to have an unlimited number of
        indicators going into another indicator as input, if this is
        what is defined in the condition definition.

        Parameters
        ----------
        req_in : str | tuple
            The requested inputs. If a string, it is assumed to be the name
            of a price series. If a tuple, it is assumed to be a description
            of another indicator.

        data : tp.Data
            an OHLCV dictionary

        max_inputs : int
            max number of inputs to be returned. This needs to be flexible,
            because some indicators may have more inputs than others.

        Returns
        -------
        tuple[tp.ArrayLike]
            a time series of prices or indicator values

        Raises
        ------
        ValueError
            if at least one of the requested inputs is neither available
            in the OHLCV data dict, nor a tuple describing an indicator

        """
        inputs = []

        logger.info("[%s] processing requested inputs: %s", level, req_in)

        for idx, i in enumerate(req_in[:max_inputs]):
            logger.debug("[%s][%s] ... input: %s (%s)", level, idx, i, type(i))

            # input is a price series
            if (isinstance(i, Operand)) and (i.type_ == OperandType.SERIES):
                logger.info("[%s][%s] ... processing price series: %s", level, idx, i)

                try:
                    inputs.append(data[i.output])
                except KeyError as err:
                    logger.error("Price series %s not found in OHLCV data", i)
                    raise ValueError(
                        f"Price series {i.output_names[0]} not in OHLCV data: {err}"
                    ) from err

            # input is another indicator
            elif isinstance(i, Operand) and (i.type_ == OperandType.INDICATOR):
                if isinstance(i, str):
                    logger.error("[%s][%s] how did this end up here? %s", level, idx, i)
                logger.debug(
                    "[%s][%s] ... processing indicator: %s", level, idx, i.name
                )

                for key in i._run_indicator(data, i.indicator, level + 1):
                    inputs.append(data.get(key))

            # this should never happen (because we already check that
            # during instantiation of the Operand class)
            else:
                logger.error("[%s][%s] ... unable to process input %s", level, idx, i)
                raise ValueError(
                    f"Invalid input type or name requested {i} ({type(i)})"
                )

        return tuple(inputs)

    def _update_names(self, u_name: str, indicators: list) -> str:
        """Update the output names and uniqe_name of the operand.

        Parameters
        ----------
        u_name : str
            unique name of the operand
        indicators : list
            all indicators (including nested ones from inputs)

        Returns
        -------
        str
            a unique_name for this operand
        """
        for elem in self.inputs:
            if isinstance(elem, OperandIndicator):
                logger.debug("Updating %s with %s", self.name, elem)
                logger.debug("%s output names: %s", elem, elem.output_names)

                self.unique_name += f"_{elem.unique_name}"

                self.output_names = tuple(
                    f"{elem_out}_{sub_elem}"
                    for elem_out in self.output_names
                    for sub_elem in elem.output_names
                )

                self.output += f"_{elem.output}"

            else:
                logger.debug("Updating %s with %s", self.name, elem)
                logger.debug("%s name: %s", elem, elem.name)

                self.unique_name += f"_{elem.name}"
                self.output_names = tuple(f"{n}_{elem.name}" for n in self.output_names)
                self.output += f"_{elem.name}"

            logger.debug(
                "\tUpdated operand: %s --> unique_name: %s", self, self.unique_name
            )

            logger.debug("\toutput names after update: %s", self.output_names)

        return u_name


@dataclass(kw_only=True)
class OperandTrigger(Operand):
    """A single operand of a condition.

    A single operand can represent either:
        - an indicator
        - a series
        - a numerical value or a boolean.

    Attributes
    ----------
    name : str
        name of the operand

    type_ : OperandType
        the type of the operand.

    run_func : Optional[Callable]
        A function to run when calling the 'run' method of this operand.

    inputs : tuple
        the inputs for the indicator, for instance 'close', this
        field only applies to indicators

    unique_name : str
        the unique name of the operand, for instance 'sma_30'

    output_names: tuple[str, ...]
        The key/column names that are added to the data dict when
        running the operand.

    output : str
         The relevant output for the indicator, for instance
         'macd_signal'. This is the same as output_names for
         most indicators, except the  ones that return multiple
         values.

    parameters : TypedDict
        parameters for the indicator, only applies to indicators

    parameter_space : Optional[Sequence[Number]]
        A range of values to use for automated optimizing of the
        parameters.
    """

    name: str
    type_: OperandType
    indicator: ind.Indicator | None = field(default=None)
    run_func: Callable | None = field(default=None)
    inputs: tuple = field(default_factory=tuple)
    unique_name: str = ""
    output_names: tuple = field(default_factory=tuple)
    output: str = ""
    parameters: dict = field(default_factory=dict)
    parameter_space: Optional[Sequence[Number]] = None
    indicators: list[ind.Indicator] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"[{self.type_}] {self.name}={self.indicator.trigger}"

    def __post_init__(self) -> None:
        # make sure, inputs is a tuple or None
        if self.inputs is not None:
            self.inputs = (
                tuple((self.inputs))
                if not isinstance(self.inputs, tuple)
                else self.inputs
            )

        self.unique_name = self.indicator.unique_name
        self.output_names = tuple(self.indicator.unique_output)
        self.output = self.output_names[0]

    @property
    def display_name(self) -> str:
        """Return the display name for the operand"""
        return self.name

    @property
    def plot_desc(self) -> dict[str, tp.Parameters]:
        """Return the plot description for the operand.

        Takes the PlotDescription as returned by all indicators for
        the operand and returns a PlotDescription for the operand,
        after combining them into one. Results depend on the type of
        the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """
        desc = self.indicator.plot_desc
        return ind.PlotDescription(
            label=self.display_name,
            is_subplot=desc.is_subplot,
            lines=desc.lines,
            triggers=desc.triggers,
            channel=desc.channel,
            hist=desc.hist,
            level="operand",
        )

    # ..........................................................................
    def update_parameters(self, params: dict[str, tp.Parameters]) -> None:
        """Update the parameters for the indicator(s).

        Parameters
        ----------
        params : dict[str, tp.Parameters]
            The new parameters, keys are the names of the indicators.
            The keys/names in this dict will be checked against the
            unique names (e.g. sma_30) of all indicators that this
            operand has and/or uses as inputs. If these match, the
            indicator will be updated with the new parameters (= the
            dictionary that is the value for this key). Keys that do
            not match any unique name will be ignored.

            .. code-block:: python
            {
                'sma_30': {'timeperiod': 28},
                'rsi_14': {'timeperiod': 7}
            }

        Note: This is meant to be used by optimizers. If you want to
        use a strategy with different parameters, it may be easier to
        create a new strategy instance with the new parameters.
        """

        logger.info("parameter update request %s for %s", params, self.unique_name)

        post_init = False

        for k in params.keys():
            logger.info("processing update request for : %s", k)

            for indicator in self.indicators:
                if k == indicator.unique_name:
                    logger.info(
                        "...updating parameters for %s with %s",
                        indicator.unique_name,
                        params[k],
                    )
                    indicator.parameters = params[k]
                    post_init = True
                else:
                    logger.warning(
                        "... %s not relevant for %s", k, indicator.unique_name
                        )

        if post_init:
            self.__post_init__()

    def run(self, data: tp.Data) -> str:
        """Run the operand on the given data.

        Parameters
        ----------
        data : tp.Data
            OHLCV dictionary

        Returns
        -------
        str
            name of the key in data that was added by this operand
        """
        data[self.unique_name] = np.full_like(
            data[self.inputs[0]],
            fill_value=self.indicator.trigger
        )
        return self.unique_name

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the operand"""
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class OperandPriceSeries(Operand):
    """A operand of a condition that represents a price series.

    Attributes
    ----------
    name : str
        name of the operand

    type_ : OperandType
        the type of the operand.

    inputs : tuple
        the inputs for the indicator, for instance 'close', this
        field only applies to indicators

    unique_name : str
        the unique name of the operand, for instance 'sma_30'

    output : str
         The relevant output for the indicator, for instance
         'macd_signal'. This is the same as output_names for
         most indicators, except the  ones that return multiple
         values.
    """

    name: str
    type_: OperandType
    inputs: tuple = field(default_factory=tuple)
    unique_name: str = ""
    output: str = ""

    def __repr__(self) -> str:
        return f"[{self.type_}] {self.name.upper()}"

    def __post_init__(self) -> None:
        # make sure, inputs is a tuple or None
        if self.inputs is not None:
            self.inputs = (
                tuple((self.inputs))
                if not isinstance(self.inputs, tuple)
                else self.inputs
            )

        self.unique_name = self.name
        self.type_ = OperandType.SERIES
        self.output_names = (self.name,)
        self.output = self.output_names[0]

    @property
    def display_name(self) -> str:
        """Return the display name for the operand."""
        return self.name

    @property
    def plot_desc(self) -> dict[str, tp.Parameters]:
        """Return the plot description for the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """
        return ind.PlotDescription(label=self.name, is_subplot=False, level="operand")

    # ..................................................................................
    def update_parameters(self, params: dict[str, tp.Parameters]) -> None:
        """Update the parameters for the indicator(s).

        Parameters
        ----------
        params : dict[str, tp.Parameters]
            The new parameters, keys are the names of the indicators.
            The keys/names in this dict will be checked against the
            unique names (e.g. sma_30) of all indicators that this
            operand has and/or uses as inputs. If these match, the
            indicator will be updated with the new parameters (= the
            dictionary that is the value for this key). Keys that do
            not match any unique name will be ignored.

            .. code-block:: python
            {
                'sma_30': {'timeperiod': 28},
                'rsi_14': {'timeperiod': 7}
            }

        Note: This is meant to be used by optimizers. If you want to
        use a strategy with different parameters, it may be easier to
        create a new strategy instance with the new parameters.
        """
        logger.info("parameter update request %s for %s", params, self.unique_name)
        raise NotImplementedError(
            "It's not possible to update parameters for a price series"
        )

    def run(self, data: tp.Data) -> str:
        """Run the operand on the given data.

        Parameters
        ----------
        data : tp.Data
            OHLCV dictionary

        Returns
        -------
        np.ndarray
            resulting array from running the operand
        """
        return self.inputs[0]

    def as_dict(self):
        """Return a dictionary representation of the operand.

        Returns
        -------
        dict[str, Any]
            a dictionary representation of the operand
        """
        return asdict(self)


# --------------------------------------------------------------------------------------
#                                    OPERAND FACTORY                                   #
# --------------------------------------------------------------------------------------
def operand_factory(op_def: OperandDefinitionT) -> Operand:
    """Factory function to create an operand from a given definition.

    The 'definition' can have different formats, depending on the type
    of the operand:
    - a tuple describing an indicator
    - a string describing an indicator or a price series (e.g. 'close')
    - a dict, describing a fixed value, e.g. {'overbought': 100}

    Parameters
    ----------
    op_def : OperandDefinitionT
        an operand description in the form of: tuple | str

    Returns
    -------
    Operand
        an Operand class, ready for use
    """
    all_indicators = []

    def from_tuple_indicator(op_def, level=0) -> OperandIndicator:
        """Build an Operand (indicator) instance from a tuple definition.

        for indicators with more than one return value, the value
        to be used for comparisons by the condition must be
        specified by appending it with dot notation to the indicator
        name, for instance 'macd.macdsignal'. Now we need to find the
        right value for the 'output' attribute here.

        Parameters
        ----------
        op_def : tuple
            the tuple describing the operand/indicator, example:
            ('rsi', 'close', {'timeperiod': 14})

        level: int
            recursion level (for logging purposes), defaults to 0

        Returns
        -------
        OperandIndicator
            an OperandIndicator instance

        Raises
        ------
        ValueError
            if the indicator name is not valid

        ValueError
            if the inputs are not valid

        ValueError
            if the output (names) cannot be determined from the indicator
        """

        splitted = op_def[0].split(".")
        name = splitted.pop(0)

        # get indicator class instance from factory
        try:
            i: ind.Indicator = copy.copy(ind.factory(name.upper()))
        except AttributeError as err:
            logger.error("indicator not found: %s (%s)", op_def[0], err)
            raise ValueError(f"invalid indicator: {op_def[0]}") from err

        all_indicators.append(i)

        logger.debug("indicator: %s", i)

        # set the indicator parameters, only if the last element in the
        # provided tuple is a dictionary
        if kwargs := op_def[-1] if isinstance(op_def[-1], dict) else {}:
            logger.debug("[%s] setting indicator parameters: %s", level, kwargs)
            i.parameters = kwargs
            inputs_pre = op_def[1:-1]
        else:
            inputs_pre = op_def[1:]

        try:
            inputs = eval_inputs(inputs_pre, i, level)
        except ValueError as err:
            logger.error("[%s] unable to evaluate inputs: %s", level, err)
            raise

        try:
            output = (
                next(filter(lambda x: x.endswith(splitted[0]), i.unique_output))
                if splitted
                else i.unique_output[0]
            )
        except StopIteration:
            logger.error(
                "unable to find output for %s in %s", splitted[0], i.unique_output
            )
            raise ValueError("unable to find output for %s in %s", i, i.unique_output)

        return OperandIndicator(
            name=i.name,
            type_=OperandType.INDICATOR,
            inputs=inputs,
            indicator=i,
            indicators=all_indicators,
            output=output,
        )

    def from_tuple_fixed(op_def: tuple) -> OperandTrigger:
        """Build an operand (trigger) from a tuple definition.

        Parameters
        ----------
        op_def : tuple
            the tuple describing the operand

        Returns
        -------
        OperandTrigger
            an OperandTrigger instance

        Raises
        ------
        ValueError
            if the tuple does not have at least 2 elements (name/value)

        ValueError
            if the value is not a bool, int or float
        """
        if len(op_def) < 2:
            raise ValueError(f"operand definition needs at least 2 elements: {op_def}")

        name, value = op_def[0], op_def[1]

        params = op_def[-1] if len(op_def) == 3 and isinstance(op_def[-1], dict) else {}
        params["value"] = value

        i = ind.factory(indicator_name=name, params=params, source="fixed")

        match value:
            case bool():
                op_type = OperandType.BOOL
            case int():
                op_type = OperandType.VALUE_INT
            case float():
                op_type = OperandType.VALUE_FLOAT
            case _:
                raise ValueError(
                    f"no valid 'value' in definition: {value} (type{type(value)})"
                )

        return OperandTrigger(
            name=name, type_=op_type, inputs=("close",), indicator=i, indicators=(i,)
        )

    def from_str(op_def: str) -> Operand:
        """Builds an Operand instance from  a string definition.

        Parameters
        ----------
        op_def : str
            the name of the indicator or price series

        Returns
        -------
        Operand
            an Operand instance

        Raises
        ------
        ValueError
            if the input name or type is not valid
        """
        if op_def in ALL_INDICATORS:
            return from_tuple_indicator(tuple((op_def,)))

        if op_def in VALID_PRICE_INPUTS:
            op_def = op_def.lower()

            return OperandPriceSeries(
                name=op_def,
                type_=OperandType.SERIES,
                inputs=(op_def,),
                output=op_def,
            )

        raise ValueError(
            f"Invalid input type or name requested: " f"{op_def} (type{type(op_def)})"
        )

    def eval_inputs(inputs_pre: Sequence, i: ind.Indicator, level: int) -> tuple:
        """Evaluate the inputs for the indicator.

        As each indicator can have inputs of different kind, including
        other indiicators, this function evaluates the inputs and - if
        necessary - builds more operands for inputs that are indicators
        or price series.

        Parameters
        ----------
        inputs_pre : Sequence
            the descriptions of the indicators as given by the client

        i : ind.Indicator
            the indicator to evaluate the inputs for

        level : int
            recursion level, for logging purposes

        Returns
        -------
        tuple
            a tuple of of one or more inputs for the indicator

        Raises
        ------
        TypeError
            if the inputs descriptions are not a string or a tuple
        """
        inputs_pre = inputs_pre or i.input
        inputs: list[Any] = []

        for idx, input_ in enumerate(inputs_pre):
            logger.debug(
                "[%s][%s] evaluating input for %s: %s", idx, level, i.name, input_
            )

            match input_:
                # input is another indicator, description given as tuple
                case tuple():
                    inputs.append(from_tuple_indicator(input_, level + 1))
                # input_ indicator or price series, given as string
                case str():
                    inputs.append(from_str(input_))
                # don't accept anything else
                case _:
                    raise TypeError(
                        f"Invalid input type or name for {i.name} "
                        f"requested: {input_} (type{input_})"
                    )

        # make sure the number of inputs matches the number of inputs
        # required by the indicator
        if (length_req := len(inputs)) > (length_max := len(i.input)):
            logger.warning(
                "%s inputs provided, but %s accepts only "
                "%s! discarding some inputs ...",
                length_req,
                i.name,
                length_max,
            )
            inputs = inputs[: len(i.input)]

        elif length_req < len(i.input):
            logger.warning(
                "%s inputs provided, but %s requires "
                "%s! filling up with default inputs...",
                length_req,
                i.name,
                length_max,
            )
            inputs.extend(i.input)
            inputs = inputs[: len(i.input)]

        return tuple(inputs)

    # ..........................................................................
    match op_def:
        # if the first string in the op_def is an indicator ..
        case tuple() if op_def[0].split(".")[0].lower() in ALL_INDICATORS:
            logger.debug("creating operand from tuple: %s", op_def)
            return from_tuple_indicator(op_def)
        # otherwise the string will be interpreted as the name of a
        # fixed value trigger
        case tuple():
            logger.debug("creating operand from tuple: %s", op_def)
            return from_tuple_fixed(op_def)
        # if the op_def is a string, it will be interpreted as the name
        # of a price series or an indicator with default parameters
        case str():
            logger.debug("creating operand from string: %s", op_def)
            return from_str(op_def)
        # try again!
        case _:
            logger.error("Unable to build operand from definition: %s", op_def)
            raise ValueError(f"Invalid operand type: {type(op_def)}")
