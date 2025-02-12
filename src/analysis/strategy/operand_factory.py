#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 03 01:03:20 2025

@author dhaneor
"""

import copy
import logging
import re
from enum import Enum, unique
from typing import Any, Sequence

from .operand import (
    Operand, OperandIndicator, OperandSeries, OperandTrigger, OperandType
    )
from ..indicators import indicator as ind
from ..indicators import indicators_custom
from analysis.models.market_data import MarketData

logger = logging.getLogger("main.operand_factory")
logger.setLevel(logging.ERROR)

ALL_INDICATORS = set(i.upper() for i in ind.talib.get_functions())
CUSTOM_INDICATORS = tuple(name.upper() for name in indicators_custom.custom_indicators)


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

VALID_PRICE_INPUTS = {member.value for member in PriceSeries}


# ======================================================================================
class OperandDefinition:
    """Class to formalize operand descriptions."""

    KNOWN_INTERVALS = [
        "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6", "8", "12h", "1d", "1w", "1M"
        ]

    def __init__(self, definition: str | tuple):
        self.name: str = ""
        self.extension: str = ""
        self.type_: OperandType = None
        self.inputs: list[str] = []
        self.params: dict[str, Any] = {}
        self.interval: str = ""
        self.shift: int = 0
        self._parameter_space: list = []

        self._parse_definition(definition)

    def __repr__(self):
        return (
            f"OperandDefinition(name='{self.name}', inputs={self.inputs}, "
            f"params={self.params}, interval='{self.interval}', shift={self.shift}, "
            f"parameter_space={self._parameter_space})"
        )

    def _parse_definition(self, definition: str | tuple):
        if isinstance(definition, str):
            self._parse_string(definition)
        elif isinstance(definition, tuple):
            self._parse_tuple(definition)
        else:
            raise ValueError(f"Invalid operand definition: {definition}")

    def _parse_string(self, definition: str):
        self.name = definition

        if definition.upper() in ALL_INDICATORS:
            self.type_ = OperandType.INDICATOR
        elif definition.upper() in CUSTOM_INDICATORS:
            self.type_ = OperandType.INDICATOR
        else:
            self.type_ = OperandType.SERIES
            self.inputs = [definition]

    def _parse_tuple(self, definition: tuple):
        self.parse_name(definition[0])

        for item in definition[1:]:
            match item:

                case str():
                    if item in self.KNOWN_INTERVALS:
                        self.interval = item
                    elif self._is_valid_input(item):
                        self.inputs.append(item)
                    else:
                        raise ValueError(f"Invalid input or interval: {item}")

                case dict():
                    self.params.update(item)

                case int():
                    if definition[-1] == item:
                        self.shift = item
                    else:
                        self.type_ = OperandType.TRIGGER
                        self.inputs.append(item)

                case float():
                    self.type_ = OperandType.TRIGGER
                    self.inputs.append(item)

                case tuple():
                    self.inputs.append(item if self._is_valid_input(item) else None)

                case list():
                    try:
                        self._is_valid_parameter_space(item)
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid parameter space: {item} -> {e}"
                            ) from e
                    else:
                        self._parameter_space = item

                case _:
                    raise ValueError(
                        f"Invalid item in operand definition: {item} ({self.type_})"
                    )

            logger.debug(f"{self.name} going to evaluate the inputs ...")
            self.inputs = self.parse_inputs(self.inputs)

    def parse_name(self, name) -> str:
        splitted_name = re.split(r"[\s\-.]+", name)
        base_name = splitted_name[0].upper()
        extension = splitted_name[1] if len(splitted_name) > 1 else None
        logger.debug(f"{self.name} going to parse the name... {base_name}")

        is_custom = base_name in CUSTOM_INDICATORS

        logger.debug(
            "%s is in custom indicators %s: %s", base_name, CUSTOM_INDICATORS, is_custom
            )

        if base_name in ALL_INDICATORS or base_name in CUSTOM_INDICATORS:
            self.type_ = OperandType.INDICATOR
        elif name in PriceSeries:
            self.type_ = OperandType.SERIES
        elif isinstance(name, str):
            self.type_ = OperandType.TRIGGER
        elif isinstance(name, int):
            self.type_ = OperandType.VALUE_INT
        elif isinstance(name, float):
            self.type_ = OperandType.VALUE_FLOAT
        elif isinstance(name, bool):
            self.type_ = OperandType.BOOL

        logger.debug("[%s] determined type: %s", base_name, self.type_)

        self.name = base_name

        if extension:
            self.extension = extension
            self.output = name

    def parse_inputs(
        self, requested: list[str, tuple], level: int = 0
    ) -> list["OperandDefinition"]:
        logger.debug(f"   [{level}] {self.name} Parsing inputs: {requested}")

        if self.type_ == OperandType.INDICATOR:
            result = []
            for input_ in requested:
                logger.debug(f"   [{level}] {self.name} input found: {input_}")
                match input_:

                    case str():
                        logger.debug("[%s] %s processing string: %s", self.name, level, input_)
                        logger.debug("[%s] %s inputs: %s", level, self.name, self.inputs)
                        if input_ in self.inputs:
                            if input_ in VALID_PRICE_INPUTS:
                                result.append(input_)
                            else:
                                result.append(OperandDefinition(input_))

                    case tuple():

                        result.append(OperandDefinition(input_))

                    case OperandDefinition():
                        result.append(input_)

                    case _:
                        raise ValueError(f"Invalid operand input: {input_}")

            return result

        elif self.type_ == OperandType.SERIES:
            return [requested[0]]

        elif self.type_ == OperandType.TRIGGER:
            return [requested[0]]

    def _is_valid_input(self, input_: str | int | float) -> bool:
        match input_:
            case int() | float():
                return True
            case str():
                return (
                    input_ in VALID_PRICE_INPUTS
                    or re.match(r"^[a-zA-Z_]\w*$", input_)
                    or input_ in ALL_INDICATORS
                    or input_ in CUSTOM_INDICATORS
                )
            case tuple():
                return True
            case _:
                raise ValueError(f"Invalid operand input: {input_}")

    def _is_valid_parameter_space(self, space: list | dict) -> bool:
        match space:
            case dict():
                return self._valid_parameter_space_dict(space)
            case list():
                return self._valid_parameter_space_list(space)
            case _:
                raise ValueError(f"Invalid operand parameter space: {space}")

    def _valid_parameter_space_dict(self, space: dict) -> dict:
        if not self.type_ == OperandType.SERIES:
            raise ValueError(
                "Operand parameter space can only be a dictionary for series"
            )

        for param, value in space.items():
            if not isinstance(param, str) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Invalid operand parameter space entry: {param}={value}"
                )

        return True  # valid dictionary of parameters for the operand

    def _valid_parameter_space_list(self, space: list) -> list:
        if not self.type_ == OperandType.TRIGGER:
            raise ValueError("Operand parameter space can only be a list for triggers")
        if not all(isinstance(item, (int, float)) for item in space):
            raise ValueError(
                "Operand parameter space can only contain integer or float values"
            )


# --------------------------------------------------------------------------------------
#                                    OPERAND FACTORY                                   #
# --------------------------------------------------------------------------------------
class Factory:
    """Class to build operands from OperandDefinition instances."""
    def __init__(self):
        self.all_indicators: list[ind.Indicator] = []
        self.id_keys: int = 0
        self.market_data: MarketData = None

    def build_operand(self, definition: OperandDefinition) -> Operand:
        self.all_indicators = []  # keep track of all indicators created

        match definition.type_:
            case OperandType.INDICATOR:
                operand = self._build_indicator(definition)
            case OperandType.SERIES:
                operand = self._build_series(definition)
            case OperandType.TRIGGER:
                operand = self._build_trigger(definition)
            case _:
                raise ValueError(f"Invalid operand type: {definition.type_}")

        # if not self.market_data:
        #     logger.error("Market data not set")
        #     raise ValueError("market data not set")

        # set market data and id in a unified manner for all
        # types of operands
        # operand.market_data = self.market_data
        operand.id = self.id_keys
        self.id_keys += 1
        
        return operand

    def _build_indicator(self, definition: OperandDefinition, level=0) -> OperandIndicator:
        logger.debug(f"[{level}] Building indicator: {definition.name}")

        # get indicator class instance from factory
        try:
            indicator = copy.copy(
                ind.factory(
                    definition.name.upper(),
                    on_change=OperandIndicator.on_parameter_change
                    )
                )
        except AttributeError as err:
            logger.error("indicator not found: %s (%s)", definition.name, err)
            raise ValueError(f"invalid indicator: {definition.name}") from err

        self.all_indicators.append(indicator)

        # set the indicator parameters, if provided
        if definition.params:
            indicator.parameters = definition.params

        inputs = self._eval_inputs(definition.inputs, indicator, level)

        # Some indicators (like the MACD) have more than one output. But
        # for one Operand, we are only interested in one of them (e.g.:
        # MACD Signal) to compare it with other Operands. That's why we
        # set the .output property of the Operand, which then is also the
        # key that is used to look up the actual data in the 'data' dict
        # when running the operands/indicators.
        # The extension (if applicable) is set during the building of the
        # OperandDefinition (see class above), so we  can extract it here.
        # if definition.extension:
        #     logger.debug(f"[{level}] Setting indicator extension: {definition.extension}")
        #     try:
        #         output = (
        #             next(
        #                 filter(
        #                     lambda x: x.endswith(definition.extension), indicator.unique_output
        #                 )
        #             )
        #         )
        #     except StopIteration:
        #         logger.error(
        #             "unable to find output for %s in %s",
        #             definition.extension, indicator.unique_output
        #         )
        #         raise ValueError(
        #             "unable to find output for %s in %s",
        #             definition.name, indicator.unique_output
        #             )
        # else:
        #     output = indicator.unique_output[0]

        # logger.debug(f"[{level}] Setting indicator output: {output}")
        logger.debug("[%s ]all indicators: %s" % (level, self.all_indicators))

        return OperandIndicator(
            name=indicator.name,
            extension=definition.extension,
            type_=OperandType.INDICATOR,
            inputs=inputs,
            indicator=indicator,
            indicators=self.all_indicators[level:],
            # output=output,
        )

    def _eval_inputs(self, inputs_pre: Sequence, i: ind.Indicator, level: int) -> tuple:
        """Evaluate the inputs for the indicator.

        As each indicator can have inputs of different kind, including
        other indicators, this function evaluates the inputs and - if
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
        logger.debug(
            "[%s] evaluating inputs (%s) for %s", level, inputs_pre, i.name
            )

        inputs_pre = inputs_pre or i.input
        inputs: list[Any] = []

        logger.debug("[%s]   indicator inputs: %s", level, i.input)

        for idx, input_ in enumerate(inputs_pre):
            logger.debug(
                "[%s][%s]   evaluating input for %s: %s", level, idx, i.name, input_
            )

            match input_:
                # input is another indicator, description given as tuple
                case OperandDefinition():
                    inputs.append(self._build_indicator(input_, level + 1))
                # input_ indicator or price series, given as string
                case str():
                    definition = OperandDefinition(input_)
                    inputs.append(self._build_series(definition))
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

        logger.debug(f"[{level}]   returning inputs for {i.name}: {inputs}")

        return tuple(inputs)

    def _build_series(self, definition: OperandDefinition) -> Operand:
        """Builds an Operand instance from a string definition.

        Parameters
        ----------
        name : str
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

        if definition.name in VALID_PRICE_INPUTS:
            return OperandSeries(
                name=definition.name.lower(),
                type_=OperandType.SERIES,
                inputs=(definition.name,),
                output=definition.name,
                market_data=self.market_data,
            )

        raise ValueError(
            f"Invalid input type or name requested: "
            f"{definition.name} (type: {definition.type_}))"
        )

    def _build_trigger(self, definition: OperandDefinition) -> OperandTrigger:
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
        # ............................................................................
        name, params = definition.name, definition.params
        value, space =definition.inputs[0], definition._parameter_space

        # determine operand type
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

        # check (and if necessary build) the param dictionary
        match space:
            case dict():
                self._check_parameter_space(name, value, space)
            case list() | tuple() | set():
                space = {name: {name: space}}
                self._check_parameter_space(name, value, space)
            case _:
                raise ValueError(f"invalid 'space' in definition: {space}")

        params['parameter_space'] = space[name]

        # add the fixed value to the parameters to comply with the format that is
        # expected by the indicator factory function
        params[name] = value

        # get indicator class instance from factory
        i = ind.factory(indicator_name=definition.name, params=definition.params, source="fixed")

        return OperandTrigger(
            name=definition.name,
            type_=op_type,
            inputs=definition.inputs,
            indicator=i,
            indicators=(i,)
        )

    def _check_parameter_space(
        self,
        name: str,
        value: int | float | bool,
        params: dict
    ) -> None:
        p_space_dict = params.get(name, {})

        if not p_space_dict:
            logger.error(f"no parameter space defined for {name}")
            logger.error(f"got parameter dictionary: {params}")
            raise ValueError(f"no parameter space defined for {name}")

        p_space_seq = p_space_dict.get(name, ())

        # make sure that the type of the value is compatible with the
        # type of the elements in the parameter space
        for elem in p_space_seq:
            if not self._is_compatible_type(value, elem):
                raise ValueError(
                    f"value {value} for {name} incompatible "
                    f"with {elem} in parameter space"
                )

        # make sure that the step size is positive
        if p_space_dict.get("step_size", 1) <= 0:
            raise ValueError(f"step size for {name} must be positive")

        # make sure that the step size is smaller than the difference between
        # the first and second elements in the parameter space. if not, set it
        # to one tenth of the difference
        if p_space_dict.get("step_size", 1) > (p_space_seq[1] - p_space_seq[0]):
            p_space_dict["step_size"] = (p_space_seq[1] - p_space_seq[0]) / 10

        # make sure that the first element in the parameter space is
        # smaller than the second one, exchange them if necessary
        if p_space_seq[0] > p_space_seq[1]:
            p_space_seq[0], p_space_seq[1] = p_space_seq[1], p_space_seq[0]

    def _is_compatible_type(self, value_1, value_2) -> bool:
        if isinstance(value_1, (int, float)):
            return True if isinstance(value_2, (int, float)) else False
        elif isinstance(value_1, bool):
            return True if isinstance(value_2, (bool, int)) else False
        else:
            raise ValueError(
                f"invalid 'value' in definition: {value_1} (type{type(value_1)})"
            )


factory = Factory()


# -------------------------------------------------------------------------------------
def operand_factory(
    operand_definition: tuple | str, 
    market_data: MarketData
) -> Operand:
    """Factory function to create an operand from a given definition.

    The 'definition' can have different formats, depending on the type
    of the operand:
    - a tuple describing an indicator
    - a string describing an indicator or a price series (e.g. 'close')
    - a dict, describing a fixed value, e.g. {'overbought': 100}

    Parameters
    ----------
    operand_definition : tuple | str
        An operand description in the form of a tuple or a string

    market_data : MarketData
        A MarketData instance that provides the OHLCV data for the operand(s)

    Returns
    -------
    Operand
        an Operand class, ready for use
    """
    logger.info("-------------- first: CREATING OPERAND DEFINITION -----------------")
    logger.debug("operand_definition: %s", operand_definition)
    
    operand_def = OperandDefinition(operand_definition)
    
    logger.info("%s" % operand_def)
    logger.debug(vars(operand_def))

    logger.info("--------------------- next: BUILDING OPERAND ----------------------")
    
    factory.market_data = market_data
    
    try:
        operand = factory.build_operand(operand_def)
    except ValueError as e:
        logger.exception(e)
        operand = None

    logger.info(operand)
    return operand
