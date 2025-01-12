#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the signal generator class and its factory.

Use the factory function to create instances of SignalGenerator!


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

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Any, Callable
from functools import reduce, wraps  # noqa: F401
import itertools
import logging
import numpy as np

from ..chart.plot_definition import SubPlot  # noqa: F401
from ..chart.tikr_charts import SignalChart  # noqa: F401
from .condition import (
    ConditionParser, ConditionDefinitionT, ConditionResult, cmp_funcs
    )
from analysis.models.market_data import MarketData
from .operand import Operand
from .operand_factory import operand_factory
from ..indicators.indicator import Indicator
from ..indicators.indicator_parameter import Parameter
from util import proj_types as tp
from util import log_execution_time, DotDict  # noqa: F401
from wrappers.base_wrapper import SignalsWrapper

logger = logging.getLogger("main.signal_generator")
logger.setLevel(logging.ERROR)


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
        if args:  #  and isinstance(args[0], SignalsDefinition):
            signals_def = args[0]

            operands = {}
            conditions = {
                "open_long": [],
                "close_long": [],
                "open_short": [],
                "close_short": [],
            }
            indicator_counters = defaultdict(int)
            indicator_names = {}

            def get_unique_operand_name(operand_value):
                if isinstance(operand_value, tuple):
                    indicator_name = operand_value[0]
                    if indicator_name in ['open', 'high', 'low', 'close', 'volume']:
                        return indicator_name

                    # Check if we've already assigned a unique name to this exact indicator
                    indicator_key = str(operand_value)
                    if indicator_key in indicator_names:
                        return indicator_names[indicator_key]

                    indicator_counters[indicator_name] += 1
                    unique_name = f"{indicator_name}_{indicator_counters[indicator_name]}"
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

                for condition_type in ["open_long", "close_long", "open_short", "close_short"]:
                    if hasattr(condition, condition_type):
                        condition_value = getattr(condition, condition_type)
                        if condition_value:
                            transformed_condition = list(condition_value)
                            for i, item in enumerate(transformed_condition):
                                if item in ["a", "b", "c", "d"]:
                                    operand_attr = f"operand_{item}"
                                    if hasattr(condition, operand_attr):
                                        operand_value = getattr(condition, operand_attr)
                                        unique_name = get_unique_operand_name(operand_value)
                                        transformed_condition[i] = unique_name
                            conditions[condition_type].append(tuple(transformed_condition))

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


class Signals(SignalsWrapper):
    """Class to hold the signals generated by a SignalGenerator."""

    def __init__(self, signals: np.ndarray, symbols: Sequence[str]):
        """Initializes the signals.

        The signals should be a 2D numpy array with the columns
        being the names of the symbols/markets.

        Parameters:
        ----------
        signals : np.ndarray
            the signals as a 2D numpy array
        columns : Sequence[str]
            the names of the operands
        """
        super().__init__(data=signals, columns=symbols)

    def apply_weight(self, weight: int | float) -> None:
        """Applies a weight to the signals.

        Parameters:
        ----------
        weight : int | float
            the weight to apply
        """
        if weight != 1:
            self.data = np.multiply(self.signals, weight)



@dataclass
class SignalGeneratorDefinition(DotDict):
    """Class to hold the definition of a SignalGenerator."""
    name: str
    operands: dict[str, tuple | str] 
    conditions: ConditionDefinitionT


class SignalGenerator:
    """A signal generator.

    Attributes:
    ----------
    name
        the name of the signal generator
    conditions
        a sequence of Condition objects
    plot_desc
        the plot description(s) for the signal generator
    """

    dict_keys: tuple[str, ...] = (
        "open_long",
        "open_short",
        "close_long",
        "close_short",
    )

    def __init__(
        self,
        name,
        operands: dict[str, Operand],
        conditions: ConditionDefinitionT,
    ):
        """Initializes the signal generator.

        The instance attributes are not set here. Use the factory
        function of this module to create instances of SignalGenerator!

        """
        self.name: str = name
        self.operands = operands
        self.conditions = conditions
        self._market_data: MarketData = None

    def __repr__(self):
        op_str = "\n  ".join(f"{k}: {v}" for k, v in self.operands.items())

        return (
            f"\nSignalGenerator: {self.name}\n"
            f"{self.conditions}\n[Operands]  \n  {op_str}\n"
            f"[KEY STORE]\n  {self.key_store}"
        )

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
        return tuple(
            itertools.chain(
                ind for op in self.operands.values() for ind in op.indicators
            )
        )

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
            the parameters used by the signal generator
        """
        return tuple(p for ind in self.indicators for p in ind.parameters)

    @parameters.setter
    def parameters(self, params: tuple[Any, ...]) -> None:
        for p_current, p_new in zip(self.parameters, params):
            try:
                p_current.value = p_new
            except Exception as e:
                logger.error(
                    "Unable to set parameter %s to %s: %s",
                    p_current.name, p_new, str(e)
                    )

    # @property
    # def subplots(self) -> list[SubPlot]:
    #     """Get the plot parameters for the signal generator.

    #     Returns
    #     -------
    #     list[SubPlot]
    #         A list of unique SubPlot objects for all conditions.
    #     """
    #     # Collect all PlotDescription objects for all conditions
    #     all_plots = [c.plot_desc for c in self.conditions]

    #     # Remove duplicates while preserving order
    #     unique_plots = []
    #     seen = set()
    #     for plot in all_plots:
    #         if plot.label not in seen:
    #             seen.add(plot.label)
    #             unique_plots.append(plot)

    #     return unique_plots

    # @log_execution_time(logger)
    def execute(self, market_data: MarketData | None= None) -> ConditionResult:
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

        # Update market_data for all operands if it is provided,
        # otherwise use the current market_data
        if market_data is not None:
            for operand in self.operands.values():
                operand.market_data = market_data

        # run the conditions for each action (open/close - long/short)
        #
        # actions = ['open_long', 'open_short', 'close_long', 'close_short']
        # or_conditions = either must be true
        # and_conditions = all must be true
        #
        # # conditions are nested lists (one for each action), where the 
        # inner layer contains conditions that must all be true (AND). 
        # the results from the sub-lists are combined using logical OR
        signals = {
            'open_long': None,
            'open_short': None,
            'close_long': None,
            'close_short': None,
        }

        for action, or_conditions in self.conditions.items():
            logger.debug(f"Running conditions for: {action}...")

            if or_conditions is None:
                continue
            
            or_result = None
            for and_conditions in or_conditions:
                
                and_result = None
                for condition in and_conditions:
                
                    left_operand = self.operands[condition[0]]
                    right_operand = self.operands[condition[2]]
                    operator = cmp_funcs[condition[1]]

                    single_result = operator(  # a 2D array
                        left_operand.run(), 
                        right_operand.run() 
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
        
            signals[action] = Signals(or_result, symbols=self.market_data.symbols)

        # .............................................................................
        return signals

    def speak(self, market_data: MarketData) -> tp.Data:
        return self.execute(market_data)

    def randomize(self) -> None:
        for operand in self.operands.values():
            operand.randomize()

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
    #         data=df, subplots=self.subplots, style=style, title=self.name
    #         )

    def is_working(self) -> bool:
        """Check if the signal generator is working.

        Returns
        -------
        bool
            True if the signal generator is working, False otherwise
        """
        raise NotImplementedError()

    def combine_signals(data):
        """
        Combine different trading signals into a single array of position indicators.

        This function takes a dictionary of trading signals and combines them into a
        single numpy array. The array uses the following convention:
        1 for long positions, -1 for short positions, 0 for closing positions,
        and NaN for no signal.

        Parameters
        ----------
        data : dict
            A dictionary containing the following keys:
            - 'open_long': Array of signals to open long positions
            - 'open_short': Array of signals to open short positions
            - 'close_long': Array of signals to close long positions
            - 'close_short': Array of signals to close short positions

        Returns
        -------
        numpy.ndarray
            An array of the same length as the input signals, where:
            - 1 indicates opening a long position
            - -1 indicates opening a short position
            - 0 indicates closing a position (either long or short)
            - NaN indicates no signal
        """
        open_long = np.nan_to_num(data["open_long"])
        open_short = np.nan_to_num(data["open_short"])
        close_long = np.nan_to_num(data["close_long"])
        close_short = np.nan_to_num(data["close_short"])

        return np.where(
            open_long > 0, 1, np.where(
                open_short > 0, -1, np.where(
                    close_long > 0, 0, np.where(
                        close_short > 0, 0, np.nan
                    )
                )
            )
        )


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
        definition.name, 
        operands, 
        condition_parser.parse(definition.conditions)
        )
