#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides Operand classes and their factory function.

NOTE:   
Operands that are indicators can have the output from another indicator 
as input. This is also the main reason level exists in the 
SignalGenerator -> Condition -> Operand -> Indicator -> Parameter chain. 
This allows for the construction of complex and flexible conditions by 
defining nested indicators without having to write a new indicator class.


classes:
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

Created on Sat Aug 18 10:356:50 2023

@author: dhaneor
"""
import logging
import numpy as np
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from numbers import Number
from typing import Any, Callable, Optional, Sequence

from ..indicators import indicator as ind
from ..indicators import indicators_custom
from ..indicators.indicator_parameter import Parameter
from ..models.market_data import MarketData
from util import proj_types as tp
from util import log_execution_time  # noqa: F401
from ..chart.plot_definition import SubPlot, Line, Channel

logger = logging.getLogger("main.operand")
logger.setLevel(logging.ERROR)

# build a list of all available indicators, that can later be used
# to do a fast check if requested indicators are available.
ALL_INDICATORS = set(i.lower() for i in ind.talib.get_functions())
CUSTOM_INDICATORS = tuple(name.lower() for name in indicators_custom.custom_indicators)

MAX_CACHE_SIZE = 100 * 1024 * 1024 # max cache size for operands


# ======================================================================================
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


# ======================================================================================
@dataclass
class Operand(ABC):
    """A single operand of a condition.

    A single operand can represent either:
        - an indicator
        - a series
        - a numerical value or a boolean.

    NOTE:   Operands that are indicators can have the output from
            another indicator as input. This is also the main reason
            level exists in the SignalGenerator -> Condition ->
            Operand -> Indicator -> Parameter chain.
            This allows for the construction of complex and flexible
            trading strategies / conditions.

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
        The unique name of the operand, for instance 'sma_30_close'.
        This name is used to identify the operand in the data
        dictionary. It ends with the name of the input series to
        distinguish the keys/columns when running the same operand/
        indicator with the same parameter for different inputs( for
        instance: highs and lows of the price/OHLCV data).

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
    interval : str = ""
    params: dict[str, Any] = field(default_factory=dict)

    parameter_instances = None

    _market_data: MarketData = None   
    _parameter_space: Optional[Sequence[Number]] = None
    _unique_name: str = ""

    shift: Parameter = None
    indicators: list[ind.Indicator] = field(default_factory=list)

    id: int = field(default=0)

    inputs: tuple = field(default_factory=tuple)
    # unique_name: str = ""
    _output: str = ""

    _cache = {}

    def __repr__(self) -> str:
        name = f"{self.name.upper()}" if isinstance(self.name, str) else f"{self.name}"

        id_str = f"[{self.id}]"

        if not self.indicator:
            return f"{id_str} {self.type_}{name}"

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
                param_str, in_str = "", ""

        return f"{id_str} {self.type_} {name}{param_str}{in_str} -> {self.unique_name}"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    def market_data(self) -> MarketData:
        return self._market_data
    
    @market_data.setter
    def market_data(self, market_data: MarketData) -> None:
        self._market_data = market_data

        for input_ in self.inputs:
            if isinstance(input_, Operand):
                input_.market_data = market_data

        self._cache.clear()

    @property
    def parameters_tuple(self) -> tuple[Any, ...]:
        """Return the parameters for the operand as a tuple"""
        return tuple(p.value for p in self.parameter_instances)

         # return tuple(chain.from_iterable(i.parameters_tuple for i in self.indicators))

    @abstractmethod
    @property
    def plot_desc(self) -> dict[str, tp.Parameters]: ...

    @abstractmethod
    def run(self, data: tp.Data) -> str: ...

    @abstractmethod
    def as_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def randomize(self) -> None:
        ...

    # @log_execution_time(logger)
    def _update_cache(self, key, value) -> None:
        """Update the cache with the current unique_name."""
        
        def get_total_size(obj):
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                return size + sum(get_total_size(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return size + sum(get_total_size(i) for i in obj)
            else:
                return size

        size_value = get_total_size(value)

        if size_value > MAX_CACHE_SIZE:
            logger.warning("[%s]   item too large to cache: %s" % (self.name, key))
            return

        cache_size = get_total_size(self._cache)

        if cache_size + size_value > MAX_CACHE_SIZE:
            # remove the least recently used item from the cache
            # (which is the first item in the dictionary)
            del self._cache[list(self._cache.keys())[0]]
            logger.warning(
                "[%s]   Cache size exceeded, removing least recently used item: %s"
                % (self.name, list(self._cache.keys())[0])
            )
        
        self._cache[key] = value

        if logger.level <= logging.DEBUG:
            cache_size = get_total_size(self._cache)

            if cache_size < 1024:
                cs_str = f"{cache_size} B"
            elif cache_size < 1024 * 1024:
                cs_str = f"{cache_size / 1024:.0f} kB"
            else:
                cs_str = f"{cache_size / (1024 * 1024):.1f} MB"

            logger.debug(
                "[%s]   updated cache. number of saved results: %s (cache size: %s)", 
                self.name, len(list(self._cache.keys())), cs_str
                )
            
            first_key = list(self._cache.keys())[0]
            logger.debug(self._cache[first_key])


@dataclass(kw_only=True)
class OperandIndicator(Operand):
    """Operand that represents an indicator."""
    name: str
    type_: OperandType

    indicator: ind.Indicator | None = field(default=None)

    inputs: tuple = field(default_factory=tuple)
    extension: str | None = None

    parameters: dict = field(default_factory=dict)

    _unique_name: str = ""
    _output_names: list = field(default_factory=list)

    run_func: Callable | None = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()

    def __post_init__(self) -> None:
        # make sure, inputs is a tuple or None
        if self.inputs is not None:
            if not isinstance(self.inputs, tuple):
                self.inputs = tuple((self.inputs))

        if not self.indicator:
            logger.error("Operand %s has no indicator", self.name)
            raise AttributeError(f"Operand {self.name} has no indicator")

        if len(self.indicator.output) > 1 and not self.extension:
            raise AttributeError(
                f"\nOperand {self.name.upper()} has multiple outputs but no name "
                "extension.\nOperands that represent this kind of indicator must have a name "
                "extension. \nExample for defining the MACD: "
                "('macd.macdsignal', {'fastperiod': 3, 'slowperiod': 15}).\n"
                f"Available extensions for {self.name.upper()}: "
                f"{', '.join(self.indicator.output)}"
                )

        self.parameters = {self.unique_name: i.parameters for i in self.indicators}

        space = {}

        for indicator in self.indicators:
            indicator.add_subscriber(self.on_parameter_change)
            space.update({indicator.name: indicator.parameter_space})

        self._parameter_space = space

        self.parameter_instances = (
            p for ind in self.indicators for p in ind.parameters
        )

    @property
    def display_name(self) -> str:
        """Return the display name for the operand"""
        return " ".join((i.display_name for i in self.indicators))

    @property
    def output(self) -> str:
        """Return the output name for the operand"""

        # Some indicators (like the MACD) have more than one output. But
        # for one Operand, we are only interested in one of them (e.g.:
        # MACD Signal) to compare it with other Operands. That's why we
        # set the .output property of the Operand, which then is also the
        # key that is used to look up the actual data in the 'data' dict
        # when running the operands/indicators.
        # The extension (if applicable) is set during the building of the
        # OperandDefinition (see class above), so we  can extract it here.
        if self.extension:
            output = f"{self.unique_name}_{self.extension}"
        else:
            output = self.unique_name
        logger.debug("[%s]   updated output: %s" % (self.name, self._output))
        return output

    @property
    def output_names(self) -> tuple:
        """Return the output names for the operand"""
        if len(self.indicators) == 1:
            self._output_names = list(self.indicator.unique_output)
            logger.debug(
                "   using original output name: %s"
                % list(self.indicator.unique_output)
                )

        else:
            for elem in self.inputs:
                if isinstance(elem, OperandIndicator):
                    logger.debug("Updating %s with %s", self.name, elem)
                    logger.debug("%s output names: %s", elem, elem._output_names)

                    self._output_names = [self.unique_name]
                    self._output_names.append(elem.unique_name)

        self._output_names = list(set(self._output_names))
        self._output_names.reverse()

        logger.debug("[%s]\toutput names after update: %s", self.name, self._output_names)

        return self._output_names

    @property
    def parameter_space(self) -> dict[dict[str: Sequence[Number]]]:
        """Return the parameter space for the operand"""
        return {i.name: i.parameter_space for i in self.indicators}

    @property
    def plot_desc(self) -> SubPlot:
        """Return the plot description for the operand.

        Takes the SubPlot object as returned by all indicators for
        the operand and returns a Subplot for the operand,
        after combining them into one. Results depend on the type of
        the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """

        def belongs_to_channel(line: Line) -> bool:
            if "upper" in line.label:
                return True
            elif "lower" in line.label:
                return True
            else:
                return False

        desc = self.indicator.plot_desc

        logger.debug(f"Operand {self.name} plot_desc:\n {desc}")

        # get the lines from the main indicator that do not
        # belong to a channel
        lines = tuple(
            elem for elem in desc.elements
            if isinstance(elem, Line) and not belongs_to_channel(elem)
        )

        for line, name in zip(lines, self.output_names):
            line.column = name

        logger.debug(f"Operand {self.name} lines: {lines}")

        # if the input for the main indicator is another indicator,
        # update label and column for the line(s)
        if len(self.indicators) > 1:
            logger.debug(f"Operand {self.name} has multiple indicators")
            lines[0].column = self.unique_name

        logger.debug(f"Operand {self.name} combined lines: {lines}")

        # combine the lines that belong to a channel into a single
        # Channel element
        channel_lines = [
            elem for elem in desc.elements
            if isinstance(elem, Line) and belongs_to_channel(elem)
        ]

        if channel_lines:
            channel = Channel(
                upper=channel_lines[0],
                lower=channel_lines[1],
                label=None,
            )
        else:
            channel = ()

        return SubPlot(
            label=" of ".join((i.plot_desc.label for i in self.indicators)),
            is_subplot=any(arg.is_subplot for arg in self.indicators),
            elements=tuple((channel, *lines)) if channel else lines,
            level="operand",
        )

    @property
    def unique_name(self) -> str:
        """Return the unique name for the operand
        
        The unique name is used to identify the operand and its data
        in the data dictionary which is used for plotting, for instance.

        This prevents potential conflicts when multiple operands use the
        same indicator with different parameters.
        """
        logger.debug("[%s]   updating unique_name" % self.name)
        logger.debug("[%s]   current unique_name: %s" % (self.name, self._unique_name))
        logger.debug("[%s]   indicators: %s" % (self.name, self.indicators))
        ind_unique = [ind.unique_name for ind in self.indicators]
        last = ind_unique.pop(-1)

        # if we only have one indicator (default case), the list we 
        # created withwe the unique names of all indicators used by 
        # this operand is empty, after removong the last element. We
        # can use the removed element to create the unique_name of
        # this indicator.
        if not ind_unique:
            self._unique_name = last
        # In case of multiple/nested indicators, we create a unique_name
        # for this operand, e.g.: 'sma_10_rsi_14_close'
        else:
            # remove the input names of the indicators from their unique_name
            splitted = [name.split("_")[:-1] for name in ind_unique]

            # join the remaining unique_names with '_' and add the last one
            # (which still has its input name attached to the end)
            self._unique_name = (
                f"{('_').join(('_'.join(elem) for elem in splitted))}_{last}"
            )

        logger.debug("[%s]   updated unique_name: %s" % (self.name, self._unique_name))
        return self._unique_name

    # .................................................................................
    # @log_execution_time(logger)
    def run(self) -> str:
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
        if (parameters := self.parameters_tuple) in self._cache:
            return self._cache[parameters]
        
        result = self._run_indicator({})
        self._update_cache(parameters, result)
        return result

    def as_dict(self):
        """Return a dictionary representation of the operand.

        Returns
        -------
        dict[str, Any]
            a dictionary representation of the operand
        """
        return asdict(self)

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

        logger.debug("parameter update request %s for %s", params, self.unique_name)

        if isinstance(params, dict):
            self._update_parameters_from_dict(params)

        self._update_names()

    def _update_parameters_from_dict(self, params: dict[str, Any]) -> None:
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

    def on_parameter_change(self) -> None:
        # logger.debug("Parameter change event for %s", self.unique_name)
        self._update_names()
        self._cache = {}

    def randomize(self) -> None:
        logger.debug("Randomizing parameters for operand %s", self.name)
        for indicator in self.indicators:
            indicator.randomize()

    # ..........................................................................
    def _run_indicator(
        self,
        data: tp.Data | None,
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

        logger.info("[%s] Running indicator: %s", level, indicator_.unique_name)

        # get the required input(s) for the indicator from the data
        logger.debug("Getting inputs for indicator: %s", self.inputs)

        inputs = self._get_ind_inputs(
            req_in=self.inputs,
            data=data,
            max_inputs=len(indicator_.input),
            level=level
        )

        logger.debug("%s inputs for indicator: %s",  len(inputs), indicator_.name)
        logger.debug("requested inputs: %s", self.inputs)

        indicator_values = indicator_.run(*inputs)
        number_of_outputs = len(indicator_values)

        if data:
            # the indicator may return a single array, or a tuple of arrays
            if number_of_outputs == 1:
                logger.debug(
                    "[%s] %s adding %s to data", level, indicator_.name, self.output
                    )
                data[self.output] = indicator_values

            else:
                data.update(
                    {
                        key: indicator_values[idx]
                        for idx, key in enumerate(self.output_names)
                    }
                )

        return indicator_values if level > 0 else indicator_values[0]
        
        # return name of relevant key that was added to the data dict
        # return self.output_names if level > 0 else self.output

    def _get_ind_inputs(
        self, req_in: str | tuple, 
        data: tp.Data, 
        max_inputs: int, 
        level: int
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

        logger.debug("[%s] processing requested inputs: %s", level, req_in)

        for idx, i in enumerate(req_in[:max_inputs]):
            logger.debug("[%s][%s] ... input: %s (%s)", level, idx, i, type(i))

            # input is a price series - the run() method of a SeriesOperand
            # will return the requested price series taken from its reference
            # to the MarketData object that all operands have access to
            if (isinstance(i, Operand)) and (i.type_ == OperandType.SERIES):
                logger.debug("[%s][%s] ... processing price series: %s", level, idx, i)

                try:
                    inputs.append(i.run())
                except KeyError as err:
                    logger.error("Price series %s not found in OHLCV data", i)
                    raise ValueError(
                        f"Price series {i.output_names[0]} not in OHLCV data: {err}"
                    ) from err

            # input is another indicator - for nested indicators, the input is
            # another OperandIndicator instance. In thie case, we call the 
            # _run_indicator() method recursively until we reach the level
            # where all inputs are one or more price/volume series
            elif isinstance(i, Operand) and (i.type_ == OperandType.INDICATOR):
                logger.debug(
                    "[%s][%s] ... processing indicator: %s", level, idx, i.name
                )

                for key in i._run_indicator(data, i.indicator, level + 1):
                    inputs.append(i.run())

            # this should never happen (because we already check that
            # during instantiation of the Operand class)
            else:
                logger.error("[%s][%s] ... unable to process input %s", level, idx, i)
                raise ValueError(
                    f"Invalid input type or name requested {i} ({type(i)})"
                )

        return inputs

    def _update_names(self) -> str:
        """Update the output names and uniqe_name of the operand."""
        raise DeprecationWarning(
            "This method is depracated. Use the peroperties for: "
            "'unique_name', 'output', 'output_names' instead."
            )


@dataclass(kw_only=True)
class OperandTrigger(Operand):
    """A single operand of a condition that represents a fixed value.

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

    # name: str
    # type_: OperandType
    indicator: ind.Indicator | None = field(default=None)
    run_func: Callable | None = field(default=None)
    inputs: tuple = field(default_factory=tuple)
    output_names: tuple = field(default_factory=tuple)
    parameters: dict = field(default_factory=dict)
    # parameter_space: Optional[Sequence[Number]] = None
    indicators: list[ind.Indicator] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"[{self.id}] {self.type_} {self.name}={self.indicator.parameters[0].value}"

    def __post_init__(self) -> None:
        # make sure, inputs is a tuple or None
        if self.inputs is not None:
            self.inputs = (
                tuple((self.inputs))
                if not isinstance(self.inputs, tuple)
                else self.inputs
            )

        self.indicator.add_subscriber(self.on_parameter_change)

        self._unique_name = self.indicator.unique_name
        self.output_names = tuple(self.indicator.unique_output)
        self.output = self.output_names[0]

        self.parameter_space = self.indicator.parameter_space

        self.parameter_instances = (p for p in self.indicator.parameters)

    @property
    def display_name(self) -> str:
        """Return the display name for the operand"""
        return self.indicator.display_name

    @property
    def unique_name(self) -> str:
        """Return the unique name for the operand"""
        # return f"{self.name}_{self.indicator.parameters[0].value}"
        return "%s_%s" % (self.name, self.indicator.parameters[0].value)

    @property
    def plot_desc(self) -> dict[str, tp.Parameters]:
        """Return the plot description for the operand.

        Takes the SubPlot as returned by all indicators for
        the operand and returns a SubPlot for the operand,
        after combining them into one. Results depend on the type of
        the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """
        desc = self.indicator.plot_desc
        return SubPlot(
            label=self.display_name,
            is_subplot=desc.is_subplot,
            elements=desc.elements,
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

    # @log_execution_time(logger)
    def run(self) -> str:
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
        
        try:
            # logger.debug("trying cache ... %s", self.name)
            return self._cache[self.unique_name]
        except KeyError:
            logger.debug("Running %s", self.name)
            self._cache[self.unique_name] = np.full_like(
                self.market_data.close,
                fill_value=self.indicator.parameters[0].value,
            )
            return self._cache[self.unique_name]

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the operand"""
        return asdict(self)

    def on_parameter_change(self) -> None:
        logger.debug("Parameter change event for %s", self.unique_name)
        self.__post_init__()
        self.inputs = [self.indicator.parameters[0].value]
        super().on_parameter_change()

    def randomize(self) -> None:
        logger.info("Randomizing parameters for %s", self.unique_name)
        self.indicator.randomize()


@dataclass(slots=True, kw_only=True)
class OperandSeries(Operand):
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
    columns: dict[str, str] = field(default_factory=dict)
    inputs: tuple = field(default_factory=tuple)
    unique_name: str = ""
    output: str = ""
    market_data: MarketData = None

    def __repr__(self) -> str:
        return f"[{self.id}] {self.type_} {self.name.upper()}"

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
    def plot_desc(self) -> SubPlot | None:
        """Return the plot description for the operand.

        Returns
        -------
        dict[str, tp.Parameters]
            plot parameters for the operand
        """
        # return SubPlot(label=self.name, is_subplot=False, level="operand")
        return None

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

    def run(self) -> str:
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
        if not self._cache:
            self._cache[self.name] = self.market_data.get_array(self.inputs[0])

        return self._cache[self.name]

    def randomize(self) -> None:
        logger.info("skipping non-randomizable operand: %s" % self.name)

    def as_dict(self):
        """Return a dictionary representation of the operand.

        Returns
        -------
        dict[str, Any]
            a dictionary representation of the operand
        """
        return asdict(self)
