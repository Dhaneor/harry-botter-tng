#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a factory for different indicators for Technical Analysis.

classes:
    Indicator
        Base class for all technical indicators.
    FixedIndicator
        Wrapper for fixed values to appear as indicator.

Functions:
    factory
        Factory function for Indicator objects.

Created on Sat Aug 05 22:39:50 2023

@author: dhaneor
"""
import inspect
import itertools
import logging
import numpy as np
import pandas as pd
import talib

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from numbers import Number
from typing import (
    Callable,
    Mapping,
    Literal,
    Sequence,
    Union,
    Dict,
    Any
)
from talib import MA_Type, abstract

from .indicator_parameter import Parameter

logger = logging.getLogger("main.indicator")
logger.setLevel(logging.INFO)

Params = Dict[str, Union[str, float, int, bool]]
IndicatorSource = Literal["talib", "nb"]

MA_TYPES = MA_Type.__dict__.get("_lookup", [])


def get_all_indicator_names() -> list[str]:
    """Returns a list of all available indicator names.

    Returns
    -------
    list[str]
        List of all available indicator names.
    """
    return talib.get_functions()


@dataclass(frozen=True)
class PlotDescription:
    """Plot description for one indicator.

    This description is used to tell the chart plotting component how
    to plot the indicator. Every indicator builds and returns this
    automatically

    Attributes
    ----------
    label: str
        the label for the indicator plot

    is_subplot: bool
        does this indicator require a subplot or is it layered with
        the OHLCV data in the main plot

    lines: Sequence[tuple[str, str]]
        name(s) of the indicator values that were returned with the
        data dict, or are then a column name in the dataframe that
        is sent to the chart plotting component

    triggers: Sequence[tuple[str, str]]
        name(s) of the trigger values, these are separated from the
        indicator values to enable a different reprentation when
        plotting

    channel: tuple[str, str]
        some indicators require plotting a channel, this makes it
        clear to the plotting component, which data series to plot
        like this.

    level: str
        the level that produced this description, just to make
        debugging easier
    """

    label: str
    is_subplot: bool
    lines: Sequence[tuple[str, str]] = field(default_factory=list)
    triggers: Sequence[tuple[str, str]] = field(default_factory=list)
    channel: list[str] = field(default_factory=list)
    hist: list[str] = field(default_factory=list)
    level: str = field(default="indicator")

    def __add__(self, other: "PlotDescription") -> "PlotDescription":
        res_levels = {
            0: "indicator",
            1: "operand",
            2: "condition",
            3: "signal generator",
        }

        # determine the 'level' of the new description after addition
        level = max(
            (
                k
                for k, v in res_levels.items()
                for candidate in (self.level, other.level)
                if v == candidate
            )
        )

        # combine the lines
        lines = list(set(itertools.chain(self.lines, other.lines)))

        # combine the triggers
        triggers = list(set(itertools.chain(self.triggers, other.triggers)))

        if triggers:
            trig_channel = [
                elem[0]
                for pair in itertools.pairwise(triggers)
                if pair[0][0].split("_")[0] == pair[1][0].split("_")[0]
                for elem in pair
            ]
        else:
            trig_channel = []

        # combine the channels ... include the triggers, so we can
        # plot a chnnel between 'overbought' and 'oversold' for example
        channel = list(set(itertools.chain(self.channel, other.channel, trig_channel)))

        # sometimes the addition can cause a line to be both a line
        # and a trigger, so we need to remove it from the lines
        lines = [line for line in lines if line not in triggers]

        return PlotDescription(
            label=self.label,
            is_subplot=self.is_subplot & other.is_subplot,
            lines=lines,
            triggers=triggers,
            channel=channel,
            level=res_levels[min(level + 1, 3)],
        )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


def get_parameter_space(param_name: str) -> dict:
    """
    Set the parameter space for an indicator if it's not already defined.

    This function iterates through the parameters of the given indicator and sets
    a default parameter space for each parameter that doesn't already have one.
    The parameter space is defined as a list of [min, max, step] values, which can
    be used for parameter optimization.

    Parameters:
    -----------
    indicator : Indicator
        The indicator object for which to set the parameter space.

    Returns:
    --------
    None
        This function modifies the indicator object in-place and
        doesn't return anything.

    Notes:
    ------
    The function sets different parameter spaces based on the parameter name:
    - 'timeperiod': [2, 200, 2]
    - 'fastperiod', 'signalperiod', 'fastk_period': [2, 100, 2]
    - 'slowperiod', 'slowk_period': [10, 200, 2]
    - Parameters ending with 'matype': [first MA type, last MA type, 1]
    - Parameters starting with 'nbdev': [0.1, 4, 0.1]
    """
    logger.debug("   determining parameter space for %s", param_name)

    # Define a dictionary to map parameter names to their spaces
    parameter_spaces = {
        "timeperiod": [1, 200, 2],
        "timeperiod1": [2, 200, 2],
        "timeperiod2": [2, 200, 2],
        "timeperiod3": [2, 200, 2],
        "fastperiod": [2, 100, 2],
        "signalperiod": [2, 100, 2],
        "fastk_period": [2, 100, 2],
        "fastd_period": [2, 100, 2],
        "slowperiod": [10, 200, 2],
        "slowk_period": [10, 200, 2],
        "slowd_period": [10, 200, 2],
        "fastlimit": [0, 10, 1],
        "slowlimit": [0, 10, 1],
        "minperiod": [1, 100, 5],
        "maxperiod": [20, 200, 5],
        "minmovestep": [1, 10, 1],
        "maxmovestep": [1, 10, 1],
        "acceleration": [1, 10, 1],
        "accelerationinitlong": [1, 10, 1],
        "accelerationlong": [1, 10, 1],
        "accelerationmaxlong": [1, 10, 1],
        "accelerationinitshort": [1, 10, 1],
        "accelerationshort": [1, 10, 1],
        "accelerationmaxshort": [1, 10, 1],
        "maximum": [1, 10, 1],
        "startvalue": [1, 10, 1],
        "offsetonreverse": [0, 1, 1],
        "vfactor": [1, 10, 1],
        "penetration": [0, 1, 1],
    }

    # Check for exact matches
    if param_name in parameter_spaces:
        return parameter_spaces[param_name]

    # Check for patterns
    if param_name.endswith("matype"):
        ma_types = list(MA_TYPES.keys())
        return [ma_types[0], ma_types[-1], 1]

    if param_name.startswith("nbdev"):
        return [0.1, 4, 0.1]

    raise ValueError(f"No parameter space for indicator: {param_name}")


# ======================================================================================
#                               Indicator classes                                      #
# ======================================================================================
class IIndicator(ABC):
    """Abstract base class for all indicators."""

    def __init__(self) -> None:
        self._name = self.__class__.__name__.lower()
        self._update_name: bool = True
        self.input: Sequence[str] = []
        self.output: Sequence[str] = []
        self.output_flags: dict

        self._plot_desc: dict

        self._apply_func: Callable
        self._parameters: dict[str, str | int | float | bool] = {}
        self._valid_params: Sequence[str] = []

        self._parameter_space: dict[str, Sequence[Number]] = {}

    @property
    def name(self) -> str:
        """Returns the name of the indicator."""
        raise NotImplementedError(
            "The name property is not implemented for this indicator."
        )

    @property
    def unique_name(self) -> str:
        """Returns a unique name for the indicator.

        Returns
        -------
        str
            Unique name for the indicator.
        """
        raise NotImplementedError(
            "The unique_name property is not implemented for this indicator."
        )

    @property
    def unique_output(self) -> tuple[str, ...]:
        """Returns a unique output name for the indicator.

        Returns
        -------
        str
            Unique output name for the indicator.
        """
        raise NotImplementedError(
            "The unique_output property is not implemented for this indicator."
        )

    @property
    def valid_params(self) -> tuple[str, ...]:
        """Returns a list of valid parameters for the indicator.

        Returns
        -------
        tuple[str]
            List of valid parameters for the indicator.
        """
        raise NotImplementedError(
            "The valid_params property is not implemented for this indicator."
        )

    @property
    def parameters(self) -> Params:
        """Returns a dictionary of parameters for the indicator.

        Returns
        -------
        dict[str, str | int | float | bool]
            Dictionary of parameters for the indicator.
        """
        raise NotImplementedError(
            "The parameters property is not implemented for this indicator."
        )

    @property
    def is_subplot(self) -> dict:
        """Returns a dictionary of plot parameters for the indicator.

        Returns
        -------
        dict
            Dictionary of plot parameters for the indicator.
        """
        return self._is_subplot

    @property
    def plot_desc(self) -> PlotDescription:
        """Returns a formal description of plot instructions."""
        raise NotImplementedError(
            "The plot_desc property is not implemented for %s", self.__class__.__name__
        )

    @abstractmethod
    def run(self, *args) -> np.ndarray:
        """Run function that returns the result of the self._apply_func.

        This skeleton method is replaced by the factory() method
        when building the indicator class.

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            Error if array has more than 2 dimensions.
        NotImplementedError
            Error if self._apply_func is not implemented.
        """

    @abstractmethod
    def help(self):
        """Prints help information (docstring) for the class.

        Can be used to have easy access to the parameters of
        each indicator.
        """


class Indicator(IIndicator):
    """A Template used by the IndicatorFactory to create indicator objects.

    The actual indicator classes are built by the factory() method below.
    Do not use on its own!
    """

    not_subplot = talib.get_function_groups()["Overlap Studies"] + ["MIN", "MAX"]

    def __init__(self) -> None:
        super().__init__()

        self._name = self.__class__.__name__.lower()
        self._update_name: bool
        self.input: Sequence[str]
        self.output: Sequence[str]

        self._apply_func: Callable
        self._parameters: dict[str, Parameter]
        self._valid_params: tuple[str]

        self._parameter_space: dict[str, Sequence[Number]]
        self._is_subplot: bool = self._name.upper() not in Indicator.not_subplot

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - {self.parameters}"

    @property
    def name(self):
        """Returns the name of the indicator."""
        return self._name

    @property
    def unique_name(self) -> str:
        """Returns a unique name for the indicator.

        Returns
        -------
        str
            Unique name for the indicator.
        """
        return (
            f"{self.name.lower()}_"
            f"{'_'.join((str(p.value) for p in self._parameters.values()))}"
        )

    @property
    def unique_output(self) -> tuple[str, ...]:
        """Returns a unique output name for the indicator.

        Returns
        -------
        str
            Unique output name for the indicator.
        """
        if len(self.output) == 1:
            return (self.unique_name,)

        return tuple(
            (
                f"{self.unique_name}_{elem}"
                if elem.endswith("band")
                else f"{self.unique_name}_{elem}"
            )
            for elem in self.output
        )

    @property
    def valid_params(self) -> tuple[str]:
        """Returns the set of valid parameters.

        These should not and can not be changed by the user/client!

        Returns
        -------
        tuple[str]
            all valid parameters for this strategy
        """
        return self._valid_params

    @property
    def parameters(self) -> Params:
        """The default parameters for the indicator.

        These parameters will be used, if the run() method is called.

        They can be changed by the user/client during run time by
        providing a new dictionary with keys/values that should be
        updated. Other keys/values remain the same. However, if keys
        in the update dictionary are not in valid_params, they will
        be ignored and a warning will be logged.

        This means that the user/client is responsible to ensure that
        this case is handled or prevented in the first place by
        comparing the keys in the update dictionary to the valid_params
        before trying to update.
        This ensures that running analysis modules keep on running, even
        if this situation occurs - it's just that the parameters are not
        changed as intended by the user/client.

        Returns
        -------
        Params: dict[str, str | float | int]
            the parameters for the indicator.
        """
        return {name: param.value for name, param in self._parameters.items()}

    @parameters.setter
    def parameters(self, params: Params) -> None:
        logger.debug(
            "setting parameters for %s: %s -> %s", self.name, self._parameters, params
        )

        for k, v in params.items():
            # if k not in self.valid_params:
            #     logger.warning(
            #         "invalid parameter '%s', allowed: %s", k, self.valid_params
            #     )
            #     continue

            if self._parameters[k].value != v:
                logger.debug("%s setting parameter %s to %s", self._name, k, v)
                self._parameters[k].value = v
                self._update_name = True
            else:
                logger.debug("parameter %s not changed", k)

        logger.debug("done setting parameters ... ")
        logger.debug("..............................................................")

    @property
    def parameter_space(self) -> dict[str, Sequence[Number]]:
        """Returns the parameter space for the indicator."""
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, update: Mapping[str, Sequence]) -> None:
        """Updates the parameter space of the indicator.

        Parameters
        ----------
        update:
            Dictionary with the new parameter space.
        """
        if not isinstance(update, Mapping):
            raise ValueError(f"parameter space must be a Mapping, not {type(update)}")

        for param, p_space in update.items():
            if param not in self.valid_params:
                logger.warning(
                    "invalid parameter '%s', allowed: %s", param, self.valid_params
                )
                continue

            if not isinstance(p_space, Sequence):
                raise ValueError(
                    f"parameter space must be a Sequence, not {type(p_space)}"
                )

            if not (2 <= len(p_space) <= 3):
                raise ValueError(
                    "Parameter space must contain 2 or 3 values, not %s" % len(p_space)
                )

            if self._parameter_space[param] != p_space:
                logger.debug("setting parameter space for '%s' to %s", param, p_space)
                self._parameter_space[param] = p_space
            else:
                logger.debug("parameter %s not changed", param)

    @property
    def plot_desc(self) -> PlotDescription:
        lines, channel, hist, count_lines = [], [], [], len(self._plot_desc.keys())

        for k, v in self._plot_desc.items():
            v = v[0]

            if "upper" in k or "lower" in k:
                channel.append(self.unique_name + "_" + k)

            elif v == "Histogram":
                hist.append(self.unique_name + "_" + k)

            else:
                if count_lines == 1:
                    lines.append(tuple((self.unique_name, v)))
                else:
                    lines.append(tuple((f"{self.unique_name}_{k}", v)))

        return PlotDescription(
            label=self.unique_name,
            is_subplot=self._is_subplot,
            lines=lines,
            channel=channel,
            hist=hist,
            level="indicator",
        )

    def run(self, *inputs) -> np.ndarray:
        """Run function that returns the result of the self._apply_func.

        This skeleton method is replaced by the factory() method
        when building the indicator class.

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            Error if array has more than 2 dimensions.
        NotImplementedError
            Error if self._apply_func is not implemented.
        """
        raise NotImplementedError()

    def help(self):
        """Prints help information (docstring) for the class.

        Can be used to have easy access to the parameters of
        each indicator.
        """
        print(self.__doc__)


class FixedIndicator(IIndicator):
    """Wrapper for fixed values to appear as indicator

    This is like a dummy class or wrapper for anything that is
    used in an Operand class and represents a fixed value. This
    is necessary to make it easier for parameter optimizers to
    handle fixed values (like overbought/oversold values for some
    indicators), because now these behave like normal indicators.
    """

    def __init__(self, name: str, value: np.flexible | bool) -> None:
        """Initializes the FixedIndicator class.

        Parameters
        ----------
        name : str
            Name of the indicator.
        value : Number | bool
            Fixed value for the indicator.
        """
        super().__init__()

        self._name: str = name
        self.trigger: Number | bool = value
        self.output: tuple[str] = (self.unique_name,)
        self.output_names: tuple[str] = (self.unique_name,)
        self._valid_params: tuple[str, ...] = "trigger", "parameter_space"
        self._parameter_space: dict[str, Sequence[np.flexible]] = {}
        self.input: tuple[str] = ("close",)
        self._is_subplot: bool = True
        self._plot_desc: dict = {"name": ["Line"]}

    def __repr__(self) -> str:
        return self.display_name

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, FixedIndicator):
            return self.trigger == other.trigger

        return False

    # ..................................................................................
    @property
    def name(self) -> str:
        """Returns the name of the indicator."""
        return self._name.lower()

    @property
    def unique_name(self) -> str:
        """Returns a unique name for the indicator.

        Returns
        -------
        str
            Unique name for the indicator.
        """
        return f"{self.name}_{self.trigger}"

    @property
    def display_name(self) -> str:
        return " ".join(str(x).capitalize() for x in self.unique_name.split("_"))

    @property
    def unique_output(self) -> tuple[str, ...]:
        """Returns a unique output name for the indicator.

        Returns
        -------
        str
            Unique output name for the indicator.
        """
        return (f"{self.name}_{self.trigger}",)

    @property
    def valid_params(self) -> tuple[str, ...]:
        """Returns a list of valid parameters for the indicator.

        Returns
        -------
        tuple[str]
            List of valid parameters for the indicator.
        """
        return self._valid_params

    @property
    def parameters(self) -> dict[str, Any]:
        """Returns a dictionary of parameters for the indicator.

        Returns
        -------
        dict[str, str | int | float | bool]
            Dictionary of parameters for the indicator.
        """
        return {"trigger": self.trigger}

    @parameters.setter
    def parameters(self, params: Params) -> None:
        """Sets the parameters for the indicator.

        Unknown parameters are ignored and a warning is logged.

        Parameters
        ----------
        params : dict[str, int | float | bool | dict]
            Dictionary of parameters for the indicator.
        """
        logger.info("setting parameters for %s -> %s", self.name, params)

        if new_parameter_space := params.pop("parameter_space", None):
            self.parameter_space = new_parameter_space

        not_gonna_happen: list[tuple[str, str]] = []

        for k, v in params.items():
            # ignore unknown parameters
            if k not in self.valid_params:
                not_gonna_happen.append((k, "unknown parameter"))
                continue

            # check that we got the correct type
            if not isinstance(v, (str, float, int, bool)):
                msg: str = f"invalid type for parameter {k} ({type(k)}"
                not_gonna_happen.append((k, msg))
                continue

            # check that the parameter is within the parameter space
            ps = self._parameter_space.get(k, None)
            logger.debug("parameter space: %s", self._parameter_space)

            if ps and not ps[0] <= v <= ps[1]:
                msg = (f"{v} is out of range ({ps[0]}, {ps[1]})",)
                not_gonna_happen.append((k, msg))
                continue

            # seems legit, set the parameter (space)
            setattr(self, k, v)
            self.output = (f"{self.name}_{v}",)
            self.output_names = (f"{self.name}_{v}",)

        # log all invalid parameters
        if not_gonna_happen:
            for elem in not_gonna_happen:
                logger.warning("... parameter '%s' not valid: %s", elem[0], elem[1])
            logger.warning("... valid parameters: %s", self.valid_params)

    @property
    def parameter_space(self) -> dict[str, Sequence[Number]]:
        """Returns the parameter space for the indicator.

        Returns
        -------
        dict[str, Sequence[Number]]
            Parameter space for the indicator.
        """
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, update: Mapping[str, Sequence] | Sequence) -> None:
        """Updates the parameter space of the indicator.

        Parameters
        ----------
        update:
            Dictionary or Sequence with the new parameter space.
            Key must always be 'value' for fixed indicators. Format
            is: {'value': [min, max, [step]]}.
        """
        if isinstance(update, Sequence):
            update = {"trigger": update}

        if not isinstance(update, Mapping):
            raise ValueError(
                f"parameter space must be a Mapping or Sequence, not {type(update)}"
            )

        if not (new_param_space := update.get("trigger", None)):
            raise ValueError(
                f"parameter space must contain a 'trigger' key, not {update.keys()}"
            )

        if not isinstance(new_param_space, Sequence):
            raise ValueError(
                f" value for parameter space must be a Sequence, "
                f"not {type(new_param_space)}"
            )

        if not (2 <= len(new_param_space) <= 3):
            raise ValueError(
                f"parameter space must be a Sequence with 2 or 3 elements, "
                f"not {len(new_param_space)}"
            )

        self._parameter_space["trigger"] = new_param_space

    @property
    def plot_desc(self) -> PlotDescription:
        return PlotDescription(
            label=self.unique_name,
            is_subplot=self._is_subplot,
            triggers=[tuple((self.unique_name, "Line"))],
            level="indicator",
        )

    # ..................................................................................
    def run(self, *inputs) -> np.ndarray:
        """Returns the result for running the indicator.

        Returns
        -------
        np.ndarray
            array with the same dimensions as the close prices of the
            data dictionary, filled with the fixed value that was set
            for the instance of this class.
        """
        return np.full_like(inputs[0], self.trigger)

    def help(self):
        """Prints help information (docstring) for the class.

        Can be used to have easy access to the parameters of
        each indicator.
        """
        return self.run.__doc__


# --------------------------------------------------------------------------------------
#                              factory functions                                       #
# --------------------------------------------------------------------------------------
def _indicator_factory_talib(func_name: str) -> Indicator:
    """
    Create and return an Indicator instance based on a TA-Lib function.

    This factory function dynamically creates an Indicator subclass for
    a given TA-Lib function. It sets up the necessary attributes and
    methods, including input and output names, parameters, and the main
    computation method.

    Parameters
    ----------
    func_name : str
        The name of the TA-Lib function to create an indicator for. This
        should match exactly with the function name in TA-Lib
        (case-insensitive).

    Returns
    -------
    Indicator
        An instance of a dynamically created Indicator subclass that
        wraps the specified TA-Lib function.

    Notes
    -----
    The created Indicator instance includes:
    - Properly set input and output names
    - A 'run' method that calls the underlying TA-Lib function
    - Dynamically generated documentation
    - Configured parameters and parameter spaces
    """
    func_name = func_name.upper()
    class_name = func_name
    talib_func = getattr(talib, class_name)
    info = talib.abstract.Function(func_name).info

    if isinstance(info["input_names"], OrderedDict):
        if "prices" in info["input_names"].keys():
            input_names = set(info["input_names"]["prices"])
        elif "price" in info["input_names"].keys():
            input_names = {info["input_names"]["price"]}
        else:
            input_names = {price for _, price in info["input_names"].items()}

    else:
        raise NotImplementedError(
            f'cannot extract input names from {type(info["input_names"])}'
        )

    if len(info["output_names"]) == 1:
        output_names = (func_name,)
    else:
        output_names = tuple((str(x) for x in info["output_names"]))  # type: ignore

    output_flags = dict(info["output_flags"])
    parameters = dict(info["parameters"])
    display_name = info["display_name"]

    # ..........................................................................
    # helper functions
    def _build_apply_func_doc_str() -> str:
        # build the function docstring dynamically
        if "pattern" not in display_name.lower():
            roll = "(rolling) " if "timeperiod" in parameters else ""
            apply_func_doc_str = (
                f"Calculates the {roll}{display_name} "
                f"({func_name.upper()}) of a time series.\n\n"
                "Use <run(*input_names, **parameters)> method to get "
                "the calculated values!\n\n"
            )

        else:
            apply_func_doc_str = (
                f"Checks for occurence of {display_name} "
                f"({func_name.upper()}) in a time series.\n\n"
                "Use <run(*input_names)> method to get "
                "the result as boolean int values!\n\n"
            )

        if info["function_flags"] is not None:
            for flag in info["function_flags"]:
                apply_func_doc_str += f"{flag}\n"

        apply_func_doc_str += "\nInput Names\n----------\n"

        for ps in input_names:
            if isinstance(ps, list):
                for elem in ps:
                    apply_func_doc_str += f"{elem}: np.ndarray\n\t{elem} prices\n"
            else:
                apply_func_doc_str += f"{ps}: np.ndarray\n\t{ps} prices\n"

        apply_func_doc_str += "\nParameters\n----------\n"

        if parameters:
            for param, dflt in info["parameters"].items():
                param = param[0] if isinstance(param, list) else param
                apply_func_doc_str += (
                    f"{param}: : np.ndarray\n\t{param}, default: {dflt}\n"
                )

                if "matype" in param:
                    apply_func_doc_str += "\n"
                    for i, name in MA_TYPES.items():
                        apply_func_doc_str += f"\t{i}: {name}\n"
        else:
            apply_func_doc_str += "none\n"

        apply_func_doc_str += "\nReturns\n-------\n"

        for name, desc in info["output_flags"].items():
            name = func_name.lower() if name == "real" else name
            desc = f"{name} values" if desc[0] == "Line" else desc
            apply_func_doc_str += f"{name}: np.ndarray\n\t{desc}\n"

        return apply_func_doc_str

    def _build_apply_func_signature():
        # build the function signature dynamically
        sig_params = []

        # positional arguments
        for name in input_names:
            if isinstance(name, str):
                sig_params.append(
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                )
            if isinstance(name, (list, tuple)):
                sig_params.extend(
                    [
                        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                        for n in name
                    ]
                )

        # keyword arguments
        sig_params.extend(
            [
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=param_info
                )
                for name, param_info in info["parameters"].items()
            ]
        )

        return inspect.Signature(parameters=sig_params)

    # ..........................................................................
    # define the run function based on the indicator requested
    def run(
        self, *inputs: tuple[np.ndarray]
    ) -> np.ndarray | tuple[np.ndarray, ...]:  # type: ignore
        logger.debug("provided data is in format %s", type(inputs))
        if type(inputs[0]) in (np.ndarray, pd.Series, pd.DataFrame):
            logger.debug("shape of data: %s", inputs[0].shape)

        # run indicator for one-dimensional array
        if isinstance(inputs[0], np.ndarray) and inputs[0].ndim == 1:  # ignore:type
            return talib_func(*inputs, **self.parameters)  # type: ignore

        # run indicator for two-dimensional array
        # NOTE: This code is not yet fully implemented, and the commented
        #       block does not work as it is! This will be necessary for
        #       running an idicator for multiple assets at once.
        if isinstance(inputs[0], np.ndarray) and inputs[0].ndim == 2:
            raise NotImplementedError()

            # for i in range(inputs[0].shape[1]):
            # out = np.empty(
            #     (inputs[0].shape[0], inputs[0].shape[1] * len(output_names))
            # )
            # res = talib_func(*inputs, **kwargs)

            # if isinstance(res, np.ndarray):
            #     out[:, 1] = res
            # else:
            #     for j in range(len(res)):
            #         out[:, i + j] = res[j]

        raise ValueError(f"indicator {func_name} only supports 1D or 2D arrays")

    run.__signature__ = _build_apply_func_signature()  # type: ignore
    run.__doc__ = _build_apply_func_doc_str()

    ind_instance = type(
        class_name,
        (Indicator,),
        {
            "__doc__": run.__doc__,
            "__init__": Indicator.__init__,
            "display_name": display_name,
            "run": run,  # types.MethodType(run, Indicator),
        },
    )()

    ind_instance.__module__ = __name__

    ind_instance.input = input_names
    ind_instance.output = output_names  # noqa: W0212
    ind_instance._plot_desc = output_flags  # noqa: W0212
    ind_instance.display_name = display_name  # noqa: W0212
    ind_instance._valid_params = tuple(k for k in parameters.keys())  # noqa: W0212
    ind_instance._parameter_space = {k: [] for k in parameters.keys()}  # noqa: W0212
    ind_instance._parameters = {}
    for k, v in parameters.items():
        parameter_space = get_parameter_space(k)
        ind_instance._parameters[k] = Parameter(
            name=k,
            initial_value=v,
            hard_min=parameter_space[0],
            hard_max=parameter_space[1],
            step=parameter_space[2] if parameter_space[2] else 1,
        )

    return ind_instance


def fixed_indicator_factory(name, params):
    """Factory function for FixedIndicator objects.

    Parameters
    ----------
    name : str
        name of the (pseudo) indicator

    params : dict
        parameters for the indicator

    Returns
    -------
    FixedIndicator
        instance of FixedIndicator class
    """
    if name is None:
        raise ValueError("name is required")

    try:
        value = params.pop("value", None)
    except AttributeError:
        value = None

    if value is None:
        raise ValueError("value is required")

    return FixedIndicator(name, value)


def set_parameter_space(indicator: Indicator) -> None:
    """
    Set the parameter space for an indicator if it's not already defined.

    This function iterates through the parameters of the given indicator and sets
    a default parameter space for each parameter that doesn't already have one.
    The parameter space is defined as a list of [min, max, step] values, which can
    be used for parameter optimization.

    Parameters:
    -----------
    indicator : Indicator
        The indicator object for which to set the parameter space.

    Returns:
    --------
    None
        This function modifies the indicator object in-place and
        doesn't return anything.

    Notes:
    ------
    The function sets different parameter spaces based on the parameter name:
    - 'timeperiod': [2, 200, 2]
    - 'fastperiod', 'signalperiod', 'fastk_period': [2, 100, 2]
    - 'slowperiod', 'slowk_period': [10, 200, 2]
    - Parameters ending with 'matype': [first MA type, last MA type, 1]
    - Parameters starting with 'nbdev': [0.1, 4, 0.1]
    """
    for param in indicator.parameters:
        logger.debug("   setting parameter space for %s", param)

        if not indicator.parameter_space.get(param, None):

            if param.startswith("timeperiod"):
                indicator.parameter_space[param] = [2, 200, 2]

            elif param in ("fastperiod", "signalperiod", "fastk_period"):
                indicator.parameter_space[param] = [2, 100, 2]

            elif param in ("slowperiod", "slowk_period"):
                indicator.parameter_space[param] = [10, 200, 2]

            elif param.endswith("matype"):
                ma_types = list(MA_TYPES.keys())
                indicator.parameter_space[param] = [ma_types[0], ma_types[-1], 1]

            elif param.startswith("nbdev"):
                indicator.parameter_space[param] = [0.1, 4, 0.1]

            else:
                pass

        logger.debug("   parameter space for %s: %s", param, indicator.parameter_space)


# --------------------------------------------------------------------------------------
cache = {i_name: _indicator_factory_talib(i_name) for i_name in talib.get_functions()}


def factory(
    indicator_name: str, params: Params | None = None, source: IndicatorSource = "talib"
) -> Indicator:
    """Creates an indicator object based on its name .

    This is the function that should solely be used for getting
    indicator objects/instances.

    Parameters
    ----------
    indicator_name : str
        name of the indicator,
    params : dict
        parameters for the indicator
    source : str
        source of the indicator (e.g. 'talib'), special case:
        "fixed" will return wrapper instance of FixedIndicator
        for fixed values (although these are not really indicators)

    Returns
    -------
    Indicator
        a concrete implementation of the Indicator

    Raises
    ------
    NotImplementedError
        if the indicator source is not supported
    """
    logger.debug("creating indicator %s (%s) from %s", indicator_name, params, source)

    if source == "talib":
        if indicator_name in cache:
            logger.debug("using cached indicator %s", indicator_name)
            ind_instance = cache[indicator_name]
            ind_instance = _indicator_factory_talib(indicator_name)
        else:
            ind_instance = _indicator_factory_talib(indicator_name)

    elif source == "fixed":
        ind_instance = fixed_indicator_factory(indicator_name, params)

    else:
        raise NotImplementedError(f"Indicator source {source} not supported.")

    # if params:
    #     for k, v in params.items():
    #         parameter_space = get_parameter_space(k)
    #         ind_instance.parameters[k] = Parameter(
    #             name=k,
    #             initial_value=v,
    #             hard_min=parameter_space[0],
    #             hard_max=parameter_space[1],
    #             step=parameter_space[2] if parameter_space[2] else 1,
    #         )

    logger.debug(ind_instance.__dict__)

    if params:
        ind_instance.parameters = params

    # if params:
    #     for k, v in params.items():
    #         logger.debug("setting parameter %s to %s", k, v)
    #         p = ind_instance._parameters[k]
    #         logger.debug(p)
    #         if k in ind_instance.parameters:
    #             ind_instance._parameters[k].value = v

    return ind_instance
