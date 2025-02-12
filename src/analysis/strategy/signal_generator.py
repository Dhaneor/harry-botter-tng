#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the signal generator class and its factory.


Classes:

SignalsDefinition (deprecated)

A formalized description of signals (long/short/close) that may
contain one or more sequence(s) of ConditionDefinition objects
which describe all the conditions that must be True to produce
a signal.

SignalsGeneratorDefinition

A formalized description of signals (long/short) that may contain one
or more sequence(s) of conditions.

SignalGenerator

This is the core of the Strategy class(es). It takes the OHLCV data
and generates the signals for opening and closing long/short positions.


Functions:

signal_generator_factory(definition: SignalGeneratorDefinition) -> SignalGenerator:

A function to create a SignalGenerator object from a
SignalsDefinition object or a dictionary of the form:


Usage:

SignalGenerator instances are not created directly. Instead, use the
factory function which builds them from a SignalGeneratorDefinition.
By using this factory, it should be al,most impossible to have a
mis-configured or non-functioning SignalGenerator.

Below is an example of how to use the SignalGeneratorDefinition:

.. code-block:: python

aroon_osc_new = SignalGeneratorDefinition(
    name="AROON OSC (noise filtered)",
    operands=dict(
        aroonosc=("aroonosc", "high", "low", {"timeperiod": 4}),
        aroon_trigger=("aroon_trigger", 1, [-5, 5, 1]),
        er=("er", {"timeperiod": 37}),
        er_trending=("er_trending", 0.21, [0.05, 0.55, 0.1]),
    ),
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
    ),
)

NOTE:
The elements of the array which is returned be the SignalGenertor.execute()
method are structured arrays, defined by the following dtype:

.. code-block:: python
        
np.dtype([
    ('open_long', np.bool_),
    ('close_long', np.bool_),
    ('open_short', np.bool_),
    ('close_short', np.bool_),
])


Decorators:

transform_signal_definition(func: Callable[..., Any]) -> Callable[..., Any]:

Provides a decorator for transforming Signalsfinition to
SignalGeneratorDefinition instances.

SignalsDefinition = the previously used way to define signals.
SignalGeneratorDefinition = the 'new' new way to define signals.

This is just for convenience and to prevent having to rewrite
existing strategies all at once.


Created on Sat Aug 18 11:14:50 2023

@author: dhaneor
"""

from functools import wraps  # noqa: F401
# import itertools
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Sequence, Any, Callable, TypeVar, Generator

from . import (
    ConditionParser, ConditionDefinitionT,
    Operand, operand_factory, comp_funcs as cmp,
)
from analysis import (  # noqa: F401
    MarketData, Indicator, Parameter,
    combine_signals, SignalStore, 
    SubPlot
) 
from analysis.chart.plot_definition import (
    Candlestick, Line, Trigger, Buy, Sell, Positions
)
from analysis.dtypes import SIGNALS_DTYPE
from misc.mixins import PlottingMixin
from util import log_execution_time, DotDict, proj_types as tp  # noqa: F401
from models.enums import COMPARISON

logger = logging.getLogger("main.signal_generator")
logger.setLevel(logging.ERROR)

WARMUP_PERIODS = 200  # number of candles to use for warmup

ConditionT = TypeVar("ConditionT", bound=tuple[str, COMPARISON, str])


# ==================Classes to define signals / the signal generator ===================
@dataclass
class SignalsDefinition:
    """Definition for how to produce entry/exit signals.

    NOTE:
    Do not use anymore! This version is deprecated and will be removed in a
    future release. Use SignalGeneratorDefinition instead!

    Attributes:
    -----------
    name: str
        the name of the signal definition
    open_long: Sequence[ConditionDefinition]
        A sequence of conditions that must all be True to open a long position.
        For convenience, intead of a sequence, a single ConditionDefinition
        can be passed. So, you can pass here:
        - Sequence[ConditionDefinition]
        - ConditionDefinition
        - dict[str, ConditionDefinition]

    open_short: Sequence[ConditionDefinition]
        a sequence of conditions that must all be True to open a short position

    close_long: Sequence[ConditionDefinition]
        a sequence of conditions that must all be True to close a long position

    close_short: Sequence[ConditionDefinition]
        a sequence of conditions that must all be True to close a short position

    reverse: bool
        just use reversed long condition for shorts, default: False
    """

    name: str = "unnamed"
    conditions: Sequence[ConditionDefinitionT] = None

    def __repr__(self):
        out = [f"SignalsDefinition: {self.name}\n"]
        [out.append(f"\t{c}\n") for c in self.conditions if c is not None]

        return "".join(out)


def transform_signal_definition(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Provides a decorator for transforming Signalsfinition to
    SignalGeneratorDefinition instances.

    SignalsDefinition = the previously used way to define signals.
    SignalGeneratorDefinition = the 'new' new way to define signals.

    This is just for convenience and to prevent having to rewrite
    existing strategies all at once.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], SignalGeneratorDefinition):
            return func(*args, **kwargs)

        if args and isinstance(args[0], SignalsDefinition):
            signals_def = args[0]

            operands = {}
            conditions = {
                "open_long": [[]],
                "close_long": [[]],
                "open_short": [[]],
                "close_short": [[]],
            }
            indicator_counters = defaultdict(int)
            indicator_names = {}

            def get_unique_operand_name(operand_value):
                if isinstance(operand_value, tuple):
                    indicator_name = operand_value[0]
                    if indicator_name in ["open", "high", "low", "close", "volume"]:
                        return indicator_name

                    # Check if we've already assigned a unique name to this exact indicator
                    indicator_key = str(operand_value)
                    if indicator_key in indicator_names:
                        return indicator_names[indicator_key]

                    indicator_counters[indicator_name] += 1
                    unique_name = (
                        f"{indicator_name}_{indicator_counters[indicator_name]}"
                    )
                    indicator_names[indicator_key] = unique_name
                    return unique_name
                return operand_value

            for condition in signals_def.conditions:
                for operand in ["a", "b", "c", "d"]:
                    operand_attr = f"operand_{operand}"
                    if hasattr(condition, operand_attr):
                        operand_value = getattr(condition, operand_attr)
                        if operand_value:
                            unique_name = get_unique_operand_name(operand_value)
                            operands[unique_name] = operand_value

                for condition_type in [
                    "open_long",
                    "close_long",
                    "open_short",
                    "close_short",
                ]:
                    if hasattr(condition, condition_type):
                        condition_value = getattr(condition, condition_type)
                        if condition_value:
                            transformed_condition = list(condition_value)
                            for i, item in enumerate(transformed_condition):
                                if item in ["a", "b", "c", "d"]:
                                    operand_attr = f"operand_{item}"
                                    if hasattr(condition, operand_attr):
                                        operand_value = getattr(condition, operand_attr)
                                        unique_name = get_unique_operand_name(
                                            operand_value
                                        )
                                        transformed_condition[i] = unique_name
                            conditions[condition_type][0].append(
                                tuple(transformed_condition)
                            )

            transformed_def = SignalGeneratorDefinition(
                name=signals_def.name,
                operands=operands,
                conditions={k: v if v else None for k, v in conditions.items()},
            )

            args = (transformed_def,) + args[1:]

        else:
            print(f"Invalid input: {func.__name__} expects a SignalsDefinition")
            print(f"Got: {args} (type: {type(args[0])})")

        return func(*args, **kwargs)

    return wrapper


@dataclass
class SignalGeneratorDefinition(DotDict):
    """Class to hold the definition of a SignalGenerator."""

    name: str
    operands: dict[str, tuple | str]
    conditions: ConditionDefinitionT

    def __post_init__(self) -> None:
        # convet conditions to list of lists if they are not already
        for k, conditions_list in self.conditions.items():
            if isinstance(conditions_list[0], tuple):
                self.conditions[k] = [conditions_list]


# =================== Classes for building the SignalGenerator Plot ====================
class Conditions:
    def __init__(self, condtions: dict[str, ConditionT]):
        self.conditions = condtions

    def __iter__(self) -> Generator[tuple[str, int, ConditionT], None, None]:
        for action, conditions_list in self.conditions.items():
            # logger.debug("conditions_list: %s" % conditions_list)
            for idx, and_conditions in enumerate(conditions_list):
                # logger.debug("[%s] and_conditions: %s" % (idx, and_conditions))
                for condition in and_conditions:
                    yield action, idx, condition


class SubPlotBuilder:

    def __init__(self, conditions: dict[str, ConditionT]) -> None:
        self._conditions = Conditions(conditions)

        self._subplots: list[SubPlot] = []

        self._plot_data = dict()
        self._condition_results: dict[tuple, np.ndarray] = dict()

    def __repr__(self) -> str:
        return "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join(
            (sp for sp in self._subplots)
        )

    @property
    def plot_data(self) -> dict[str, np.ndarray]:
        for operand in self.operands.values():
            operand.run()  # just in case they were not run before
            self._plot_data.update(operand.plot_data)
        return self._plot_data

    @property
    def subplots(self) -> list[SubPlot]:
        if self._subplots:
            return self._subplots

        if not self._plot_data:
            self.plot_data  # call the property to fill the ._plot_data dict

        self._subplots: list[SubPlot] = [
            SubPlot(
                label="OHLCV",
                is_subplot=False,
                elements=(Candlestick(),),
                level="operand",
            )
        ]
        self._add_positions_to_ohlcv_subplot(self._subplots[0])

        unused_operands = set(self.operands.keys())        
        # go through all the operands and determine which conditions use them
        for operand_name in self.operands.keys():
            # find all conditions (by action) which use the operand
            using_this_operand = set()
            for action, idx, condition in self._conditions:
                if condition[0] == operand_name:
                    using_this_operand.add((action, idx, condition))
                    unused_operands.discard(operand_name)
                    unused_operands.discard(condition[2])
            if using_this_operand:
                self._process_operand(operand_name, using_this_operand)

        self._process_unused_operands(unused_operands)
        self._add_signal_subplot()
        return self._subplots

    def _process_operand(self, key: str, conditions: set[str, int, ConditionT]):
            logger.debug(f"Processing operand: {key} - used in conditions: {conditions}")
            operand = self.operands[key]

            # in most cases, the operand will have two subplots: OHLCV
            # and the indicator subplots ...
            if len(operand.subplots) > 1:
                subplot = operand.subplots[1]
                elements = list(subplot.elements)
            # ... but if the operand is price series, it will have only
            # the OHLCV subplot, which also contains no elements, at
            # least none that we need here
            else:
                subplot = operand.subplots[0]
                elements = []

            # Each condition tuple will contain an index which describes 
            # the level of the condition wrt the OR conditions. The final
            # plot should contain subplots that allow to distinguish
            # between different OR conditions, even when that means to
            # have two subplots for the same operand
            or_levels = max((c[1] for c in conditions)) + 1
            
            for idx in range(or_levels):
                logger.debug("processing level [%s] of OR conditions" % idx)
                # find all operands on the right side of the conditions 
                # found in the previous step and combine the elements 
                # from the SubPlots of both
                for condition in conditions:
                    add_operand_name = condition[2][2]
                    add_operand = self.operands[add_operand_name]
                    add_operand_subplots = add_operand.subplots
                    add_subplot = add_operand_subplots[1] \
                        if len(add_operand_subplots) > 1 else add_operand.subplots[0]
                    
                    existing_elements = [elem.label for elem in elements]
                    logger.debug(f"existing elements: {existing_elements}")
                    
                    for elem in add_subplot.elements:
                        if elem.label not in existing_elements:
                            logger.debug(f"adding element {elem.label} to elements")
                            elements.append(elem)

                    # execute the conditions for this operand and add their 
                    # result as element to the SubPlot for this operand
                    self._process_condition(condition, elements, operand.unique_name)
                
                # add the subplot the subplots list
                logger.debug("adding subplot to subplots list") 
                self._subplots.append(
                    SubPlot(
                        label=f"[{idx}] {subplot.label}",
                        is_subplot=subplot.is_subplot,
                        level="signal generator",
                        elements=elements
                    )
                )
                logger.debug("subplots list: %s" % [sp.label for sp in self._subplots])

    def _process_condition(self, condition: ConditionT, elements: list, label: str) -> None:
        logger.debug(f"processing condition: {condition}")
        action, idx, condition = condition
        signals = self._execute_condition(condition).astype(np.float32)

        mask = signals != 0
        try:
            signals[mask] = self._plot_data[label][mask]
            signals[~mask] = np.nan
        except KeyError as e:
            logger.error("KeyError: %s" % str(e))
            logger.error("existing keys: %s" % list(self._plot_data.keys()))
            logger.error("Exiting function due to error")

        key = f"[{idx}].{label}.{action}"
        self._plot_data[key] = signals

        match action:
            case "open_long":
                elem = Buy(label="open long", column=key, legend=None)
            case "close_long":
                elem = Sell(label="close long", column=key, legend=None)
            case "open_short":
                elem = Sell(label="open short", column=key, legend=None)
            case "close_short":
                elem = Buy(label="close short", column=key, legend=None)
            case _:
                raise ValueError(f"Invalid action: {action}")
            
        elements.append(elem)

    def _execute_condition(self,  condition: ConditionT) -> np.ndarray:
        left = self.operands[condition[0]].run()
        right = self.operands[condition[2]].run()
        func = self.cmp_funcs.get(condition[1])
        return func(left, right).reshape(-1,)[WARMUP_PERIODS:]
    
    def _add_positions_to_ohlcv_subplot(self, ohlcv_subplot: SubPlot) -> None:
        self._plot_data["position"] = self\
            .execute(compact=True)\
            .reshape(-1,)[WARMUP_PERIODS-1:-1]

        elements = list(ohlcv_subplot.elements)
        elements.insert(0, Positions())
        ohlcv_subplot.elements = elements

    def _process_unused_operands(self, unused_operands: set[str]) -> None:
        for operand_name in unused_operands:
            logger.debug(f"Adding unused operand: {operand_name}")
            operand = self.operands[operand_name]
            logger.debug("runninng operand: %s" % operand_name)
            operand.run()
            logger.debug("done")
            self._plot_data.update(operand.plot_data)
            self._subplots.append(operand.subplots[1])

    def _add_signal_subplot(self) -> None:
        raw_signal = self\
            .execute(compact=True)\
            .reshape(-1,)[WARMUP_PERIODS-1:-1]
        
        scaled_signal = np.multiply(
            raw_signal,
            self.market_data.mds.signal_scale_factor.reshape(-1,)[WARMUP_PERIODS-1:-1]
        )

        self._plot_data.update(
            {
                "signal_scaled": scaled_signal,
                "signal_zero": np.zeros_like(self.market_data.close.shape[0])
            }
        )
        self.subplots.append(
            SubPlot(
                label="signal",
                is_subplot=True,
                level="signal_generator",
                elements=(
                    Line(
                        label="signal",
                        column="signal_scaled",
                        legend=True,
                        legendgroup="Signal",
                        shape="hv"
                    ),
                    Trigger(
                        label="no signal",
                        column="signal_zero",
                        legend=False,
                        legendgroup="Signal",
                    ),
                )
            )
        )

    def _add_sub_plot(self, subplot: SubPlot) -> None:
        if subplot.label not in (sp.label for sp in self._subplots):
            self._subplots.append(subplot)


# ================================= The Signal Generator ===============================
class SignalGenerator(SubPlotBuilder, PlottingMixin):
    """A signal generator.

    Always use the factory function to create instances of 
    SignalGenerator! Refer to the documentation at the top of the 
    module for more details.

    Attributes:
    ----------
    name
        the name of the signal generator
    operands
        Operand instances (indicators, etc.)
    conditions
        a sequence of Condition definitions

    Properties:
    ----------
    indicators: tuple[Indicator]
        All Indicator instances associated with the signal generator, read-only
    market_data: MarketData
        OHLCV data in the form of a MarketData instance
    parameters: tuple[Parameter]
        All Parameter instances associated with the signal generator, read-only
    parameter_values: tuple[int | float, bool, ...]
        read/set the values of all parameters of the signal generator

    Methods:
    --------
    execute(...) -> Signals
        Runs the signal generator on the given market data (or the market
        data set at instantiation) and returns a Numpy array of signals.
        For more information about the format of the returned signals, 
        refer to the method docstring below.

    randomize() -> None
        Randomizes the (indicator) parameters of the signal generator.

    """

    cmp_funcs = {  # functions to use for comparing values of operand
        COMPARISON.IS_ABOVE: cmp.is_above,
        COMPARISON.IS_ABOVE_OR_EQUAL: cmp.is_above_or_equal,
        COMPARISON.IS_BELOW: cmp.is_below,
        COMPARISON.IS_BELOW_OR_EQUAL: cmp.is_below_or_equal,
        COMPARISON.IS_EQUAL: cmp.is_equal,
        COMPARISON.IS_NOT_EQUAL: cmp.is_not_equal,
        COMPARISON.CROSSED_ABOVE: cmp.crossed_above,
        COMPARISON.CROSSED_BELOW: cmp.crossed_below,
    }

    def __init__(
        self,
        name: str,
        operands: dict[str, Operand],
        conditions: ConditionDefinitionT,
    ):
        """Initializes the signal generator.

        The instance attributes are not set here. Use the factory
        function of this module to create instances of SignalGenerator!

        """
        SubPlotBuilder.__init__(self, conditions)
        
        self.name: str = name
        self.operands = operands
        self.conditions = conditions

        self.display_name = name

        self._indicators: tuple[Indicator] = None
        self._market_data: MarketData = None
        self._parameters: tuple[Parameter] = None

    def __repr__(self):
        op_str = "\n  ".join(f"{k}: {v}" for k, v in self.operands.items())

        return (
            f"\nSignalGenerator: {self.name}\n"
            f"{self.conditions}\n[Operands]  \n  {op_str}\n"
        )

    # ................................. PROPERTIES .....................................
    @property
    def indicators(self) -> tuple[Indicator]:
        """Get the indicator(s) used by the signal generator.

        This method should return a tuple of Indicator objects that are
        used in the conditions used by this SignalGenerator instance. It
        is used to optimize the strategy that uses this SignalGenerator.

        Returns
        -------
        tuple[Indicator]
            the indicator(s) used by the signal generator
        """
        if not self._indicators:
            self._indicators = tuple(
                chain(i for op in self.operands.values() for i in op.indicators)
            )
        return self._indicators

    @property
    def market_data(self) -> MarketData:
        """Get the market data used by the signal generator.

        Returns
        -------
        MarketData
            the market data used by the signal generator
        """
        return self._market_data

    @market_data.setter
    def market_data(self, market_data: MarketData) -> None:
        if market_data and not isinstance(market_data, MarketData):
            raise TypeError("market_data must be an instance of MarketData")
        
        if len(market_data) <= WARMUP_PERIODS:
            raise ValueError(
                f"Market data must have more than {WARMUP_PERIODS} periods, but"
                F"provided market data has {len(market_data)} periods."
            )

        self._market_data = market_data
        
        for operand in self.operands.values():
            operand.market_data = market_data

        logger.info(f"SignalGenerator {self.name} has been assigned market data")

    @property
    def parameters(self) -> tuple[Parameter]:
        """Get the parameters used by the signal generator.

        Returns
        -------
        tuple[Parameter]
            the parameters (Parameter instances) used by the signal generator
        """
        if not self._parameters:
            self._parameters = tuple(
                p for ind in self.indicators for p in ind.parameters
            )
        return self._parameters

    @property
    def parameter_values(self) -> tp.ParameterValuesT:
        """Get the current parameter values.

        Returns
        -------
        tp.ParameterValuesT
            the current parameter values
        """
        return tuple(p.value for p in self.parameters)

    @parameter_values.setter
    def parameter_values(self, params: tp.ParameterValuesT) -> None:
        """Set the parameter values.

        Parameters
        ----------
        params : tuple[int | float | bool,...]
            A tuple of new parameter values
        """
        for parameter, new in zip(self.parameters, params):
            try:
                parameter.value = new
            except Exception as e:
                logger.error(
                    "Unable to set %s to %s: %s", parameter.name, new, str(e)
                )

    # ................................ PUBLIC METHODS ..................................
    # @log_execution_time(logger)
    def execute(
        self,
        market_data: MarketData | None = None,
        param_combinations: tp.ParameterValuesT | None = None,
        compact: bool = False,
    ) -> tuple[tp.Array_3D]:
        """
        Generate signals based on market data and/or (multiple) parameter combinations.

        Parameters:
        -----------
        market_data
            The market data array (OHLCV), optional
            if not provided, use the current market_data (atrribute/property).
        param_combinations
            parameter combinations to test. If not provided, use the 
            current parameters set in the indicators.
        compact: bool, optional
            If set to True, the values will be combined into a single 
            column, with values: -1/0/1 (dtype=np.float32). This does 
            not affect the shape of the outout array, only the dtype 
            of its elements. This is the format required by the 
            backtest framework.
            If set to False, the elements of the returned array are 
            structured arrays defined by the dtype described below.
            This format is required for plotting the signals

            .. code-block:: python
            np.dtype([
                ('open_long', np.bool_),
                ('close_long', np.bool_),
                ('open_short', np.bool_),
                ('close_short', np.bool_),
                ('combined', np.float32),
            ])

        Returns:
        --------
        np.ndarray 
            A 3D array of generated signals.
            If param_combinations is None, shape is (n_timestamps, n_assets, 1).
            Otherwise, shape is (n_timestamps, n_assets, n_param_combinations).
        """

        if market_data is not None:
            self.market_data = market_data

        out = self._build_results_array(depth=len(param_combinations or [1]))

        if param_combinations is None:
            # this is the default case, execute for the current parameters
            self._execute_single(out, 0)
        else:
            # this is used during optimization, or for strategies which
            # are configured to use multiple parameter combinations - 
            # execute for each parameter combination
            for idx, params in enumerate(param_combinations):
                self.parameter_values = params
                self._execute_single(out, idx)

        return combine_signals(out) if compact else out
        
    def randomize(self) -> None:
        for operand in self.operands.values():
            operand.randomize()

    def is_working(self) -> bool:
        """Check if the signal generator is working.

        Returns
        -------
        bool
            True if the signal generator is working, False otherwise
        """
        raise NotImplementedError()
    
    # ................................. HELPER METHODS .................................
    def _execute_single(self, out: np.ndarray, zindex=0) -> dict[str, tp.Array_3D]:
        """Execute the signal generator.

        Parameters
        ----------
        data : tp.Data
            OHLCV data dictionary

        Returns
        -------
        tp.Data
            OHLCV data dictionary
        """

        # run the conditions for each action (open/close - long/short)
        #
        # actions = ['open_long', 'open_short', 'close_long', 'close_short']
        # or_conditions = either must be true
        # and_conditions = all must be true
        #
        # # conditions are nested lists (one for each action), where the
        # inner layer contains conditions that must all be true (AND).
        # the results from the sub-lists are combined using logical OR
        for action, or_conditions in self.conditions.items():

            if or_conditions is None:
                continue

            or_result = None
            for and_conditions in or_conditions:
                and_result = None
                for condition in and_conditions:
                    left_operand = self.operands[condition[0]]
                    right_operand = self.operands[condition[2]]
                    operator = self.cmp_funcs[condition[1]]

                    single_result = operator(  # a 2D array
                        left_operand.run(), right_operand.run()
                    )

                    # combine the AND condition results into a single result
                    if and_result is not None:
                        and_result = np.logical_and(and_result, single_result)
                    else:
                        and_result = single_result

                # combine the OR condition results into a single result
                if or_result is not None:
                    or_result = np.logical_or(or_result, and_result)
                else:
                    or_result = and_result

            out[action][:, :, zindex] = or_result  # .reshape(

        return out
    
    def _build_results_array(self, depth: int) -> np.ndarray:
        periods = len(self.market_data)
        assets = self.market_data.number_of_assets
        return np.zeros((periods, assets, depth), dtype=SIGNALS_DTYPE)

    # def plot(self, data: tp.Data) -> None:
    #     self.make_plot(data).draw()

    # def make_plot(self, data: tp.Data, style='night') -> SignalChart:
    #     # run the signal generator and convert the result
    #     # to a pandas DataFrame
    #     df = pd.DataFrame.from_dict(self.execute(data))

    #     # set open time to datetime format and set it as index
    #     df['open time'] = pd.to_datetime(df['open time'], unit='ms')
    #     df.set_index('open time', inplace=True)
    #     df.index = df.index.strftime('%Y-%m-%d %X')

    #     return SignalChart(
    #         data=df, subplots=self._subplots, style=style, title=self.name
    #         )


# ======================================================================================
@transform_signal_definition
def signal_generator_factory(definition: SignalGeneratorDefinition) -> SignalGenerator:
    """Build a SignalGenerator from a SignalGeneratorDefinition.

    Parameters
    ----------
    definition : SignalGeneratorDefinition
        the signal generator definition

    Returns
    -------
    SignalGenerator
        the signal generator
    """

    operands = dict()

    for name, op_def in definition.operands.items():
        operand = operand_factory(op_def, None)
        operand.id = name
        logger.info(operand)
        operands[name] = operand

    condition_parser = ConditionParser(operands)

    return SignalGenerator(
        definition.name, operands, condition_parser.parse(definition.conditions)
    )
