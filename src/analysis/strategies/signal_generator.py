#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the signal generator class and its factory.

Use the factory function to create instances of SignalGenerator!

classes:
    SignalsDefinition
        A formalized descrption of signals (long/short/close) that may
        contain one or more sequence(s) of ConditionDefinition objects
        which describe all the conditions that must be True to produce a
        signal.

    SignalGenerator
        This is the core of the Strategy class(es). It takes the
        OHLCV data and generates the signals for opening and closing
        long/short positions.

functions:
    factory(sig_def: SignalsDefinition | dict[str, cnd.ConditionDefinition]
        function to create a SignalGenerator object from a
        SignalsDefinition object or a dictionary of the form:

    .. code-block:: python
    {
        "open_long": [ConditionDefinition, ...] | None,
        "open_short": [ConditionDefinition,...] | None,
        "close_long": [ConditionDefinition,...] | None,
        "close_short": [ConditionDefinition,...] | None,
        "reverse": bool
    }


classes:
    SignalsDefinition
        formal description of signals (long/short/close)

    SignalGenerator
        the signal generator class

Functions:
    factory
        factory function for SignalGenerator


Created on Sat Aug 18 11:14:50 2023

@author: dhaneor
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Literal
from functools import reduce
import itertools
import logging
import numpy as np

from ..util import proj_types as tp
from . import condition as cnd
from ..indicators.indicator import PlotDescription, Indicator
from ..indicators.indicator_parameter import Parameter

logger = logging.getLogger("main.signal_generator")
logger.setLevel(logging.ERROR)

PositionTypeT = Literal["open_long", "open_short", "close_long", "close_short"]

ConditionDefinitionsT = Optional[
    Sequence[cnd.ConditionDefinition] | cnd.ConditionDefinition
]


@dataclass
class SignalsDefinition:
    """Definition for how to produce entry/exit signals.

    Attributes:
    -----------
    name: str
        the name of the signal definition
    open_long: ConditionDefinitionsT
        A sequence of conditions that must all be True to open a long position.
        For convenience, intead of a sequence, a single ConditionDefinition
        can be passed. So, you can pass here:
        - Sequence[cnd.ConditionDefinition]
        - cnd.ConditionDefinition
        - dict[PositionTypeT, cnd.ConditionDefinition]

    open_short: ConditionDefinitionsT
        a sequence of conditions that must all be True to open a short position

    close_long: ConditionDefinitionsT
        a sequence of conditions that must all be True to close a long position

    close_short: ConditionDefinitionsT
        a sequence of conditions that must all be True to close a short position

    reverse: bool
        just use reversed long condition for shorts, default: False
    """
    name: str = "unnamed"
    conditions: ConditionDefinitionsT = None

    def __repr__(self):
        out = [f"SignalsDefinition: {self.name}\n"]
        [out.append(f"\t{c}\n") for c in self.conditions if c is not None]

        return "".join(out)


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

    def __init__(self, name, conditions: Sequence[cnd.Condition]):
        """Initializes the signal generator.

        The instance attributes are not set here. Use the factory
        function of this module to create instances of SignalGenerator!

        """
        self.name: str = name
        self.conditions: Sequence[cnd.Condition] = conditions
        self.conditions_definitions: Sequence[cnd.ConditionDefinition] | None = None

    def __repr__(self):
        return (
            f"SignalsGenerator(name={self.name}\n"
            f"\t\t{self.conditions}\n"
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
                ind for cond in self.conditions for ind in cond.indicators
            )
        )

    @property
    def parameters(self) -> tuple[Parameter]:
        """Get the parameters used by the signal generator.

        Returns
        -------
        tuple[Parameter]
            the parameters used by the signal generator
        """
        return tuple(p for ind in self.indicators for p in ind.parameters)

    @property
    def plot_desc(self) -> tuple[PlotDescription, Sequence[PlotDescription]]:
        """Get the plot parameters for the signal generator.

        Returns
        -------
        tuple[PlotDescription, Sequence[PlotDescription]]
            the plot defintion(s), one for the main plot and one for
            each subplot
        """
        # collect all PlotDescription objects for all conditions used by
        # this SignalGenerator instance
        all_ = tuple(c.plot_desc for c in self.conditions)

        # add all PlotDescription instances that describe something that
        # goes into the main plot (= where the candlesticks are)
        main = sum(p_desc for p_desc in all_ if not p_desc.is_subplot)

        # gather all subplot definitions
        sub_with_duplicates = tuple(p_desc for p_desc in all_ if p_desc.is_subplot)

        # Some of the PlotDescription instances we collected belong
        # together, but come from different branches (for instance
        # conditions that use the same indicator but different triggers
        # to open/close a long or a short). We only want to plot the
        # indicator once, so we  need to combine those into a single
        # plot definition.
        sub = []

        # add all definitions that belong together, some of those will
        # still be duplicates
        for p_desc in sub_with_duplicates:
            sub.append(sum(p for p in sub_with_duplicates if p.label == p_desc.label))

        # remove duplicates from the result
        sub = [sub[x] for x in range(len(sub)) if not (sub[x] in sub[:x])]

        return [main].extend(sub)
        return main, sub

    def execute(self, data: tp.Data) -> cnd.ConditionResult:
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
        return reduce(
            lambda x, y: x & y,
            (cond.execute(data) for cond in self.conditions)
        )

    def speak(self, data: tp.Data, weight: tp.Weight = 1) -> tp.Data:
        return self.execute(data, weight)

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
def factory(
    sig_def: SignalsDefinition | Sequence[cnd.ConditionDefinition]
) -> SignalGenerator:
    """Factory function for SignalGenerator.

    Parameters
    ----------
    sig_def
        the signal definition or just a sequence of ConditionDefinition(s)

    Returns
    -------
    SignalGenerator
        the signal generator
    """
    match sig_def:

        case SignalsDefinition():
            logger.debug("got a SignalsDefinition instance")
            name = sig_def.name
            condition_definitions = sig_def.conditions

            # make sure we have a sequence, the elements will be validated
            # later in the condition factory function
            condition_definitions = (
                tuple((condition_definitions,))
                if isinstance(condition_definitions, cnd.ConditionDefinition)
                else condition_definitions
            )

        case tuple() | list():
            name = "Unnnamed SignalGenerator"
            condition_definitions = sig_def

        case _:
            raise TypeError(
                f"SignalsDefinition expected, got {type(sig_def)}: {sig_def}"
            )

    logger.debug("Creating signal generator for %s (%s)", sig_def, type(sig_def))
    logger.debug(
        "... Processing %s condition definition(s) ...", len(condition_definitions),
    )
    sig_gen = SignalGenerator(
        name,
        tuple(cnd.factory(c) for c in condition_definitions)
        )
    sig_gen.name = name
    sig_gen.condition_definitions = condition_definitions

    return sig_gen
