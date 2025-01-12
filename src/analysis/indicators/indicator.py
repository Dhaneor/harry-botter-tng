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
import logging
import numpy as np
import pandas as pd
import talib

from collections import OrderedDict
from typing import (
    Literal,
    Generator,
    Tuple,
    Union,
    Any,
    Callable
)
from talib import MA_Type, abstract

from .iindicator import IIndicator
from .indicators_custom import custom_indicators
from .indicator_parameter import Parameter
from ..chart.plot_definition import Line, SubPlot

logger = logging.getLogger("main.indicator")
logger.setLevel(logging.DEBUG)

Params = dict[str, Union[str, float, int, bool]]
IndicatorSource = Literal["talib", "nb"]

MA_TYPES = MA_Type.__dict__.get("_lookup", [])
Combinations = Generator[Tuple[Any, ...], None, None]

TALIB_INDICATORS = talib.get_functions()


def get_parameter_space(param_name: str) -> dict:
    """
    Get the parameter space for an indicator if it's not already defined.

    Returns the parameter space based on the name of the parameter.
    The parameter space is defined as a list of [min, max, step]
    values, which can be used for parameter optimization.

    Parameters:
    -----------
    param_name : str
        The name of the parameter.

    Returns:
    --------
    list
        A parameter space for the given parameter.

    Notes:
    ------
    The function sets different parameter spaces based on the parameter name:
    - 'timeperiod': [2, 200, 5]
    - 'fastperiod', 'signalperiod', 'fastk_period': [2, 100, 5]
    - 'slowperiod', 'slowk_period': [10, 200, 2]
    - Parameters ending with 'matype': [first MA type, last MA type, 1]
    - Parameters starting with 'nbdev': [0.1, 4, 0.1]
    """
    logger.debug("determining parameter space for %s", param_name)

    # Check for patterns
    if param_name.endswith("matype"):
        ma_types = list(MA_TYPES.keys())
        return [ma_types[0], ma_types[-1], 1]

    if param_name.startswith("nbdev"):
        return [0.25, 4, 0.25]

    # Define a dictionary to map parameter names to their spaces
    parameter_spaces = {
        "timeperiod": [2, 200, 5],
        "timeperiod1": [2, 200, 5],
        "timeperiod2": [2, 200, 5],
        "timeperiod3": [2, 200, 5],
        "fastperiod": [2, 100, 5],
        "signalperiod": [2, 100, 5],
        "fastk_period": [2, 100, 5],
        "fastd_period": [2, 100, 5],
        "slowperiod": [10, 200, 2],
        "slowk_period": [10, 200, 2],
        "slowd_period": [10, 200, 2],
        "fastlimit": [0, 10, 1],
        "slowlimit": [0, 10, 1],
        "min": [2, 120, 5],
        "max": [2, 120, 5],
        "minperiod": [2, 100, 5],
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

    raise ValueError(f"No parameter space for parameter: {param_name}")


# ======================================================================================
#                               Indicator classes                                      #
# ======================================================================================
class Indicator(IIndicator):
    """A Template used by the IndicatorFactory to create indicator objects.

    The actual indicator classes are built by the factory() method below.
    Do not use on its own!
    """

    not_subplot = talib.get_function_groups()["Overlap Studies"] + ["MIN", "MAX"]

    def __init__(self) -> None:
        super().__init__()
        self._is_subplot: bool = self._name.upper() not in Indicator.not_subplot
        self._plot_desc: dict[str, Tuple[str, Any]] = OrderedDict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - {self.parameters}"

    @property
    def plot_desc(self) -> SubPlot:
        elements = []
        idx = 0

        logger.debug(self.__dict__)

        for k, v in self._plot_desc.items():
            v = v[0]

            logger.info(f"Adding {k} to plot description: {v}")

            line = Line(
                label=f"{self.unique_name.upper()} {k if k != 'real' else ''}",
                column=self.unique_output[idx],
                end_marker=False
                )
            elements.append(line)

            logger.debug(f"Line added: {line}")

            idx += 1

        return SubPlot(
            label=f"{self.display_name} ({self.parameters[0].value})",
            elements=elements,
            is_subplot=self.is_subplot,
        )

    # def run(self, *inputs: tuple[np.ndarray]) -> np.ndarray:  # type: ignore

    #     logger.debug("provided data is in format %s", type(inputs))
    #     logger.debug("parameters: %s", self.parameters_dict)

    #     rows, columns = inputs[0].shape
    #     dimensions = inputs[0].ndim

    #     if type(inputs[0]) in (np.ndarray, pd.Series, pd.DataFrame):
    #         logger.debug("shape of data: %s", inputs[0].shape)

    #     # Apply the indicator to the inputs. We need different ways
    #     # of handling this, depending on the dimensions of the input 
    #     # data.
    #     match dimensions:
            
    #         # run indicator for one-dimensional array
    #         case 1:
    #             result = self._apply_func(inputs, **self.parameters_dict)  
            
    #         # run indicator for two-dimensional array
    #         case 2:
    #             # idicators can have one or more inputt and one
    #             # or more outputs. each case must be handled in
    #             # a different way.
    #             logger.debug("number of outputs: %s", len(self.output))
                
    #             out = [
    #                 np.full_like(inputs[0], fill_value=np.nan, dtype=np.float64)
    #                 for _ in range(len(self.output))
    #             ]

    #             logger.debug("number of result arrays: %s", len(out))
  
    #             for i in range(columns):
    #                 single_in = [
    #                     elem[:, i].reshape((rows)).astype(np.float64) 
    #                     for elem in inputs   
    #                 ]

    #                 logger.debug(
    #                     "input array (%s) has dimension: %s", 
    #                     type(single_in[0]), single_in[0].shape
    #                     )

    #                 result = self._apply_func(
    #                     *single_in,
    #                     **self.parameters_dict
    #                 )


    #                 if isinstance(result, list | tuple):
    #                     logger.debug("we got multiple output arrays from the indicator")
    #                     for j, result_elem in enumerate(result):
    #                         out[j][:, i] = result_elem
    #                     return out
    #                 else:
    #                     logger.debug("we got one output array from the indicator")
    #                     out[0][:, i] = result
    #                     logger.debug(result)
    #                     return out
            
    #         # raise a ValueError for all other cases/dimensionalities
    #         case _:
    #             raise ValueError("Unsupported array dimensions: %s" % dimensions)


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

    def __init__(self, name: str, parameter: Parameter) -> None:
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
        self.input: tuple[str] = tuple()
        self._is_subplot: bool = True
        self._plot_desc: dict = {"name": ["Line"]}

        self._data = None

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
    def plot_desc(self) -> SubPlot:
        return SubPlot(
            label=self.unique_name,
            is_subplot=self._is_subplot,
            elements=[
                Line(
                    label=self.unique_name,
                    column=self.unique_output[0],
                    end_marker=False
                )
            ],
            level="indicator",
        )

    # ..................................................................................
    def run(self, *inputs) -> np.ndarray:
        """Returns the result for running the indicator.

        Returns
        -------
        np.ndarray
            Array with shape (n, 1) where n is the length of the input array's first dimension,
            filled with the fixed value that was set for the instance of this class.
        """
        input_array = inputs[0]
        value = self._parameters[0].value

        if self._data is None or self._data.shape[0] != input_array.shape[0]:
            # Initialize or update self._data
            self._data = np.full((input_array.shape[0], 1), value, dtype=np.float32)
        elif not np.can_cast(self._data.dtype, input_array.dtype, casting='same_kind'):
            # Update dtype if necessary
            self._data = self._data.astype(input_array.dtype)

        return self._data

    def on_parameter_change(self, *args) -> None:
        super().on_parameter_change(*args)
        self._data = None

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

    # instesd of telling the indicators which inputs they should
    # use, every time we run them, these values are hard-coded here
    # (but cann be overridden at runtime)
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

    # .................................................................................
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

    # .................................................................................
    # define the run function based on the indicator requested
    def apply_func(self, *inputs: tuple[np.ndarray], **kwargs) -> np.ndarray:  # type: ignore
        return talib_func(*inputs, **self.parameters_dict)  # type: ignore

    apply_func.__signature__ = _build_apply_func_signature()  # type: ignore
    apply_func.__doc__ = _build_apply_func_doc_str()

    ind_instance = type(
        func_name,
        (Indicator,),
        {
            "__doc__": apply_func.__doc__,
            "__init__": Indicator.__init__,
            "display_name": display_name,
            "_apply_func": apply_func,  # types.MethodType(run, Indicator),
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
            "parameter_space['value'] must be a list/tuple with at least two values, "
            f"but was provided: {space}"
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

    ind = FixedIndicator(name, _parameter)

    return ind


def _custom_indicator_factory(name, params):
    """Factory function for CustomIndicator objects.

    Parameters
    ----------
    name : str
        name of the custom indicator

    params : dict
        parameters for the indicator

    Returns
    -------
    CustomIndicator
        instance of CustomIndicator class
    """
    logger.debug("Creating CustomIndicator %s -> %s", name, params)

    if name is None:
        raise ValueError("name is required")

    if not isinstance(name, str):
        raise ValueError("name must be a string")

    ind = custom_indicators.get(name)()

    if params:
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        ind.parameters = params

        space_dict = params.get("parameter_space")

        if space_dict and not isinstance(space_dict, dict):
            raise ValueError("parameter_space must be a dictionary")

    return ind


# --------------------------------------------------------------------------------------
def factory(
    indicator_name: str,
    params: Params | None = None,
    source: IndicatorSource | None = None,
    on_change: Callable | None = None,
) -> Indicator:
    """Creates an indicator object based on its name .

    This is the function that should solely be used for getting
    indicator objects/instances.

    Parameters
    ----------
    indicator_name : str
        name of the indicator,
    on_change : Callable
        A function that will be called when the indicator value changes.
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
    logger.debug(
        "creating indicator %s (params: %s) from %s",
        indicator_name, params, f" from {source}" if source else ""
        )

    if (source == "talib") | (indicator_name in talib.get_functions()):
        ind_instance = _indicator_factory_talib(indicator_name)

    elif (source == "custom") | (indicator_name.upper() in custom_indicators):
        ind_instance = _custom_indicator_factory(indicator_name, params)

    elif source == "fixed":
        ind_instance = fixed_indicator_factory(indicator_name, params)

    else:
        raise NotImplementedError(
            f"Indicator {indicator_name} not found/supported."
            f"Available sources: talib, custom")

    ind_instance.on_change = on_change

    for param in ind_instance._parameters:
        param.add_subscriber(ind_instance.on_parameter_change)

    logger.debug(ind_instance.__dict__)

    if params:
        ind_instance.parameters = params

    return ind_instance
