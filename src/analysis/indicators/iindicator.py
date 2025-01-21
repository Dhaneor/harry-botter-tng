#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the abstract Indicator class.

This abstract class is the basis for different types of indicators.

classes:
    IIndicator
        Base class for all technical indicators.
    PlotDescription
        A formalized description for the plot module.

Created on Wed Nov 06 17:35:50 2024

@author: dhaneor
"""
import logging
import numpy as np
import pandas as pd

from abc import abstractmethod
from numbers import Number
from typing import Sequence, Callable, Union, Generator, Mapping, Any

from .indicator_parameter import Parameter
from analysis.chart.plot_definition import SubPlot, Line, Channel, Histogram
from misc.mixins import PlottingMixin

logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.ERROR)

# define Types
Params = dict[str, Union[str, float, int, bool]]
Combinations = Generator[tuple[Any, ...], None, None]


# @dataclass(frozen=True)
# class PlotDescription:
#     """Plot description for one indicator.

#     This description is used to tell the chart plotting component how
#     to plot the indicator. Every indicator builds and returns this
#     automatically

#     Attributes
#     ----------
#     label: str
#         the label for the indicator plot

#     is_subplot: bool
#         does this indicator require a subplot or is it layered with
#         the OHLCV data in the main plot

#     lines: Sequence[tuple[str, str]]
#         name(s) of the indicator values that were returned with the
#         data dict, or are then a column name in the dataframe that
#         is sent to the chart plotting component

#     triggers: Sequence[tuple[str, str]]
#         name(s) of the trigger values, these are separated from the
#         indicator values to enable a different reprentation when
#         plotting

#     channel: tuple[str, str]
#         some indicators require plotting a channel, this makes it
#         clear to the plotting component, which data series to plot
#         like this.

#     level: str
#         the level that produced this description, just to make
#         debugging easier
#     """

#     label: str
#     is_subplot: bool
#     lines: Sequence[tuple[str, str]] = field(default_factory=list)
#     triggers: Sequence[tuple[str, str]] = field(default_factory=list)
#     channel: list[str] = field(default_factory=list)
#     hist: list[str] = field(default_factory=list)
#     level: str = field(default="indicator")

#     def __add__(self, other: "PlotDescription") -> "PlotDescription":
#         res_levels = {
#             0: "indicator",
#             1: "operand",
#             2: "condition",
#             3: "signal generator",
#         }

#         # determine the 'level' of the new description after addition
#         level = max(
#             (
#                 k
#                 for k, v in res_levels.items()
#                 for candidate in (self.level, other.level)
#                 if v == candidate
#             )
#         )

#         elements = itertools.chain(self.elements, other.elements)

#         # combine the lines
#         lines = list(set(elem for elem in elements if not elem.is_triger))

#         # combine the triggers
#         triggers = list(set(elem for elem in elements if not elem.is_triger))

#         if triggers:
#             trig_channel = [
#                 elem[0]
#                 for pair in itertools.pairwise(triggers)
#                 if pair[0][0].split("_")[0] == pair[1][0].split("_")[0]
#                 for elem in pair
#             ]
#         else:
#             trig_channel = []

#         # combine the channels ... include the triggers, so we can
#         # plot a chnnel between 'overbought' and 'oversold' for example
#         channel = list(set(itertools.chain(self.channel, other.channel, trig_channel)))

#         # sometimes the addition can cause a line to be both a line
#         # and a trigger, so we need to remove it from the lines
#         lines = [line for line in lines if line not in triggers]

#         return PlotDescription(
#             label=self.label,
#             is_subplot=self.is_subplot & other.is_subplot,
#             lines=lines,
#             triggers=triggers,
#             channel=channel,
#             level=res_levels[min(level + 1, 3)],
#         )

#     def __radd__(self, other: "PlotDescription") -> "PlotDescription":
#         if other == 0:
#             return self
#         else:
#             return self.__add__(other)


class IIndicator(PlottingMixin):
    """Base class for all indicators."""

    def __init__(self) -> None:
        self._name = self.__class__.__name__.lower()
        self.display_name = self._name.replace("_", " ").title()
        self.input: Sequence[str] = []
        self.output: Sequence[str] = []
        self.output_flags: dict

        self._is_subplot: bool | None = True
        self._plot_desc: dict  # e.g = {"name": ["Line"]}

        self._apply_func: Callable
        self._parameters: tuple[Parameter, ...] = ()

        self.on_change: Callable = None
        self.subscribers: set = set()

        self._cache = None

    def __repr__(self) -> str:
        return f"{self.name} - {self.parameters}"

    # ................................ PROPERTIES .....................................
    @property
    def plot_data(self) -> dict[str, np.ndarray]:
        if self._cache is None:
            raise ValueError(
                f"No data available for {self.unique_name} - "
                "Did you run the indicator before trying to access the data?"
                )
        
        data = self._cache        
        data = [data] if isinstance(data, np.ndarray) else data

        data_dict = {}
        for idx in range(len(data)):
            try:
                data_dict[self.unique_output[idx]] = data[idx]
            except Exception as e:
                logger.error(f"Error processing data for {self.unique_name}: {e}")
                logger.error(f"Data: {data}")
                raise

        return data_dict

    @property
    def name(self) -> str:
        """Returns the name of the indicator."""
        return self._name

    # @property
    # def display_name(self) -> str:
    #     return " ".join((str(x).capitalize() for x in self.unique_name.split("_")))

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
            f"_{"_".join((i for i in self.input))}"
        )

    @property
    def unique_output(self) -> tuple[str, ...]:
        """Returns a tuple with unique outputs for the indicator.

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
    def parameters_tuple(self) -> tuple[Parameter]:
        """Returns a tuple of parameter values for the indicator.

        Returns
        -------
        tuple[Parameter]
            Parameters for the indicator.
        """
        return tuple(p.value for p in self._parameters)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params: Params | tuple[int | float | bool]) -> None:
        """
        Sets the parameters for the indicator.

        This method allows setting the parameters of the indicator either
        from a dictionary or a tuple. It updates the internal parameters
        of the indicator accordingly.

        Parameters
        ----------
        params : Params | tuple
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
    def parameter_combinations(self) -> Generator:
        return self._generate_combinations(self.parameters)

    @property
    def is_subplot(self) -> dict:
        """Returns a dictionary of plot parameters for the indicator.

        Returns
        -------
        dict
            Dictionary of plot parameters for the indicator.
        """
        if self._is_subplot is None:
            raise AttributeError(
                "The ._is_subplot attribute is not set for %s", self.name
            )
        return self._is_subplot

    @property
    def subplots(self) -> tuple[SubPlot]:
        """Return the subplots for the indicator."""
        param_ext = ",".join(str(p.value) for p in self.parameters)
        subplot_label = f"{self.display_name} ({param_ext})"  # ({self.parameters[0].value})"
        channel_elements = {"upper": None, "lower": None}
        elements = []

        logger.debug(self._plot_desc)

        idx = 0
        for k, v in self._plot_desc.items():
            v = v[0]  # it's always a list with one element
            elem = None

            logger.info(f"processing: '{k}' ...")

            # talib specifies the plotting descriptions (output_flags)
            # in an OrderedDict, where the key is:
            # • 'real' for indicators with one output
            # • the name of the output for indicators with multiple outputs
            if k == "real":
                label = legend = self.name.upper() + f" ({param_ext})"
            else:
                label = legend = k.upper()
            
            legendgroup = self.name.upper()

            # the values in the output_flags dictioanry describe the 
            # type of line (or histogram for instance) to be plotted 
            # for each output
            match v:
                case "Line":
                    elem = Line(
                        label=label,
                        column=self.unique_output[idx],
                        legend=legend,
                        legendgroup=legendgroup,
                        shadow=False,
                        end_marker=False
                    )
                case "Dashed Line":
                    elem = Line(
                        label=label,
                        column=self.unique_output[idx],
                        legend=legend,
                        legendgroup=legendgroup,
                    )
                case "Histogram":
                    elem = Histogram(
                        label=label,
                        column=self.unique_output[idx],
                        legend=legend,
                        legendgroup=legendgroup,
                    )
                case "Values represent an upper limit":
                    logger.debug(
                        "[CHANNEL ELEMENTS] Adding upper limit: %s" % legend
                        )
                    channel_elements["upper"] = Line(
                        label=label,
                        column=self.unique_output[idx],
                        legend=legend,
                        legendgroup=legendgroup,
                    )
                case "Values represent a lower limit":
                    logger.debug(
                        "[CHANNEL ELEMENTS] Adding lower limit: %s" % legend
                        )
                    channel_elements["lower"] = Line(
                        label=label,
                        column=self.unique_output[idx],
                        legend=legend,
                        legendgroup=legendgroup,
                    )
                case _:
                    raise NotImplementedError(
                        f"Unsupported output_flag: {v}"
                    )
                
            if elem:                
                elements.append(elem)
                logger.debug(f"added element: {elem}")
            
            idx += 1

        # if channel elements are present, prepare a Channel
        # class and add it to the elements of the subplot
        if any(channel_elements.values()):
            if not all(channel_elements.values()):
                raise ValueError(
                    "Either upper or lower limit lines are missing. "
                    f"channel elements: {channel_elements}"
                )

            logger.debug(f"Adding channel elements: {channel_elements}")
            elem = Channel(
                label=self.name.upper(),
                upper=channel_elements["upper"],
                lower=channel_elements["lower"],
            )
            elem.shadow = False
            elements.append(elem)
            logger.debug(f"added element: {elem}")

        return SubPlot(
            label=subplot_label,
            elements=elements,
            level="indicator",
            is_subplot=self.is_subplot,
        ),

# .............................. Public methods .................................
    def run(self, *inputs: tuple[np.ndarray]) -> np.ndarray:  # type: ignore

        # if self._cache is not None:
        #     return self._cache

        logger.debug("provided data is in format %s", type(inputs))
        logger.debug("parameters: %s", self.parameters_dict)

        rows, columns = inputs[0].shape
        dimensions = inputs[0].ndim

        if type(inputs[0]) in (np.ndarray, pd.Series, pd.DataFrame):
            logger.debug("shape of data: %s", inputs[0].shape)

        # Apply the indicator to the inputs. We need different ways
        # of handling this, depending on the dimensions of the input 
        # data.
        match dimensions:
            
            # run indicator for one-dimensional array
            case 1:
                out = self._apply_func(inputs, **self.parameters_dict) 
                self._cache = out  
            
            # run indicator for two-dimensional array
            case 2:
                # indicators can have one or more inputs and one
                # or more outputs. each case must be handled in
                # a different way.
                # logger.debug("number of outputs: %s", len(self.output))
                
                out = [
                    np.full_like(inputs[0], fill_value=np.nan, dtype=np.float64)
                    for _ in range(len(self.output))
                ]

                # logger.debug("number of result arrays: %s", len(out))
  
                for i in range(columns):
                    single_in = [
                        elem[:, i].reshape((rows)).astype(np.float64) 
                        for elem in inputs   
                    ]

                    # logger.debug("-" *  120)
                    # logger.debug(
                    #     "input array (%s) has dimension: %s", 
                    #     type(single_in[0]), single_in[0].shape
                    #     )

                    result = self._apply_func(*single_in, **self.parameters_dict)
                    self._cache = result

                    if isinstance(result, list | tuple):
                        for j, result_elem in enumerate(result):
                            out[j][:, i] = result_elem
                    else:
                        out[0][:, i] = result
                                
            # raise a ValueError for all other cases/dimensionalities
            case _:
                raise ValueError("Unsupported array dimensions: %s" % dimensions)

        return out

    def help(self):
        """Prints help information (docstring) for the class.

        Can be used to have easy access to the parameters of
        each indicator.
        """
        print(self._apply_func.__doc__)
    
    def randomize(self) -> None:
        """Randomizes the parameters of the indicator."""
        logger.info("randomizing parameters for %s", self.name)
        logger.debug("Randomizing parameters: %s", self.parameters)
        for param in self.parameters:
            param.randomize()

    def add_subscriber(self, callback: Callable) -> None:
        self.subscribers.add(callback)

    def on_parameter_change(self, *args) -> None:
        """Callback function for when parameters change."""
        logger.info("parameters changed for %s", self.name)
        logger.debug("Calling subscribers: %s", self.subscribers)

        self._cache = None

        for callback in self.subscribers:
            logger.debug("Calling callback %s", callback)
            callback()

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
            param._value = params[idx]

    def _generate_combinations(self, parameters: tuple[Parameter]) -> Combinations:
        """Generates all possible combinations of elements from the given iterables.

        Yields:
            A tuple representing a combination of elements, one from each iterable.
        """
        if not parameters:
            yield ()

        first, *rest = parameters
        for item in first:
            for combination in self._generate_combinations(rest):
                yield (item,) + combination
