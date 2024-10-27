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
    Generator,
    Tuple,
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
Combinations = Generator[Tuple[Any, ...], None, None]


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
    - 'timeperiod': [1, 200, 5]
    - 'fastperiod', 'signalperiod', 'fastk_period': [1, 100, 5]
    - 'slowperiod', 'slowk_period': [10, 200, 2]
    - Parameters ending with 'matype': [first MA type, last MA type, 1]
    - Parameters starting with 'nbdev': [0.1, 4, 0.1]
    """
    logger.debug("   determining parameter space for %s", param_name)

    # Define a dictionary to map parameter names to their spaces
    parameter_spaces = {
        "timeperiod": [1, 200, 5],
        "timeperiod1": [1, 200, 5],
        "timeperiod2": [1, 200, 5],
        "timeperiod3": [1, 200, 5],
        "fastperiod": [1, 100, 5],
        "signalperiod": [1, 100, 5],
        "fastk_period": [1, 100, 5],
        "fastd_period": [1, 100, 5],
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
        return [0.25, 4, 0.25]

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
        self._parameters: Tuple[Parameter, ...] = ()

    @property
    def name(self) -> str:
        """Returns the name of the indicator."""
        return self._name

    @property
    def display_name(self) -> str:
        return " ".join((str(x).capitalize() for x in self.unique_name.split("_")))

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
            f"{'_'.join((str(p.value) for p in self._parameters))}"
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
    def valid_params(self) -> tuple[str, ...]:
        """Returns a list of valid parameters for the indicator.

        Returns
        -------
        tuple[str]
            List of valid parameters for the indicator.
        """
        return tuple(p.name for p in self.parameters)

    @property
    def parameters(self) -> Tuple[Parameter]:
        """Returns a tuple of parameters for the indicator.

        Returns
        -------
        Tuple[Parameter]
            Parameters for the indicator.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, params: Params | Tuple) -> None:
        """
        Sets the parameters for the indicator.

        This method allows setting the parameters of the indicator either
        from a dictionary or a tuple. It updates the internal parameters
        of the indicator accordingly.

        Parameters
        ----------
        params : Params | Tuple
            The parameters to set for the indicator. This can be either a
            dictionary where keys are parameter names and values are the
            parameter values, or a tuple containing parameter values in
            the order they are defined. The parameters (in order) can be
            retreived by getting the valid_params property.

        Raises
        ------
        ValueError
            If the provided params is neither a dictionary nor a tuple.
        """
        if isinstance(params, dict):
            self._set_parameters_from_dict(params)
        elif isinstance(params, tuple):
            self._set_parameters_from_tuple(params)
        else:
            raise ValueError("parameters must be either Params or a tuple")

    @property
    def parameters_dict(self) -> dict:
        """Returns the parameters of the indicator as a dictionary."""
        return {p.name: p.value for p in self.parameters}

    @property
    def parameter_space(self) -> dict[str, Sequence[Number]]:
        """Returns the parameter space for the indicator."""
        return {p.name: p.space for p in self.parameters}

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

            for p in self._parameters:
                if p.name == param:
                    p.space = p_space

    @property
    def parameter_combinations(self) -> Combinations:
        return self._generate_combinations(self.parameters)

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

    # .............................. Private methods .................................
    def _set_parameters_from_dict(self, params: dict) -> None:
        logger.info("setting parameters for %s", self.name)
        for p, v in params.items():
            if p == "parameter_space":
                # self.parameter_space = v
                break
            for param in self._parameters:
                logger.info("... setting parameter %s -> %s", param.name, v)
                if param.name == p:
                    param.value = v
                    break
            else:
                raise ValueError(f"Unknown parameter: {p}")

    def _set_parameters_from_tuple(self, params: tuple) -> None:
        logger.info("setting parameters for %s", self.name)
        for idx, param in enumerate(self._parameters):
            logger.info("... setting parameter %s -> %s", param.name, params[idx])
            param.value = params[idx]

    def _generate_combinations(self, parameters: Tuple[Parameter]) -> Combinations:
        """Generates all possible combinations of elements from the given iterables.

        Yields:
            A tuple representing a combination of elements, one from each iterable.
        """
        if not parameters:
            yield ()
        else:
            first, *rest = parameters
            for item in first:
                for combination in self._generate_combinations(rest):
                    yield (item,) + combination


class Indicator(IIndicator):
    """A Template used by the IndicatorFactory to create indicator objects.

    The actual indicator classes are built by the factory() method below.
    Do not use on its own!
    """

    not_subplot = talib.get_function_groups()["Overlap Studies"] + ["MIN", "MAX"]

    def __init__(self) -> None:
        super().__init__()
        self._is_subplot: bool = self._name.upper() not in Indicator.not_subplot

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - {self.parameters}"

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

    def __init__(
        self, name: str, value: np.flexible | bool, parameter: Parameter
    ) -> None:
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
        self._parameters = parameter,
        self.output: tuple[str] = (self.unique_name,)
        self.output_names: tuple[str] = (self.unique_name,)
        self.input: tuple[str] = ("close",)
        self._is_subplot: bool = True
        self._plot_desc: dict = {"name": ["Line"]}

    def __repr__(self) -> str:
        return self.display_name

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, FixedIndicator):
            return self._parameters[0].value == other._parameters[0].value

        return False

    # ..................................................................................
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
        return np.full_like(inputs[0], self._parameters[0].value)

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

    We are doing some class contruction magic here to automatically
    create an indicator class for the specified TA-Lib function.

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
    - Configured parameters and the (valid) parameter space
    """
    func_name = func_name.upper()
    talib_func = getattr(talib, func_name)
    info = abstract.Function(func_name).info

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
            return talib_func(*inputs, **self.parameters_dict)  # type: ignore

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
        func_name,
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

    # set the parameters & their parameter space for the indicator
    _parameters = []
    for k, v in parameters.items():
        parameter_space = get_parameter_space(k)
        _parameters.append(
                Parameter(
                    name=k,
                    initial_value=v,
                    hard_min=parameter_space[0],
                    hard_max=parameter_space[1],
                    step=parameter_space[2] if parameter_space[2] else 1,
                    )
                )
    ind_instance._parameters = tuple(_parameters)  # noqa: W0212
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
    logger.debug("Creating FixedIndicator %s -> %s", name, params)

    if name is None:
        raise ValueError("name is required")

    if not isinstance(name, str):
        raise ValueError("name must be a string")

    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    if name not in params:
        raise ValueError(f"value for {name} is required")

    # set the parameter space
    if "parameter_space" not in params:
        raise ValueError(
            "parameter_space is required for fixed indicators / triggers"
            )

    space_dict = params["parameter_space"]

    if not isinstance(space_dict, dict):
        raise ValueError("parameter_space must be a dictionary")

    if name not in space_dict:
        raise ValueError(f"parameter_space[{name}] is required")

    space = space_dict[name]

    if not isinstance(space, (list, tuple)) or len(space) < 2:
        raise ValueError(
            "parameter_space['value'] must be a list/tuple with at least two values"
        )

    try:
        min_val, max_val, = space[0], space[1]
        step = 1 if len(space) < 3 else space[2]
    except (TypeError, IndexError):
        raise ValueError(
            "parameter_space['value'] must be a list with at least two values"
            )

    _parameter = Parameter(
        name=name,
        initial_value=params[name],
        hard_min=min_val,
        hard_max=max_val,
        step=step
    )

    ind = FixedIndicator(name, params[name], _parameter)

    return ind


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

    return ind_instance
