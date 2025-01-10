#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides tools to build executable strategies dynamically.

functions:
    build_strategy(sdef: StrategyDefinition) -> IStrategy:
        main entry point for the module, should be the only function
        that a user/client needs

    build_strategy_single(sdef: StrategyDefinition) -> SubStrategy:
        Builds a single strategy.

        This is a helper function for build_strategy(). Because
        there are single and composite strategies, this function
        is specifically used for building single strategies.
        Single strategies are the elements for the sub_strategies
        attribute of a composite strategy. From a user perspective,
        there are only composite strategies, even if they contain
        only one sub-strategy.


classes:
    COMPARISON
        enum representing trigger conditions.

    ActionDefinition
        a complete set of ConditionDefinition objects for defining
        entry/exit signals for long and/or short trades. These are
        encapsulated here, so later on we can just call its
        execute() method whenever new data is available.

    StrategyDefinition
        formal description of a strategy,
        with an ActionDefinition and other parameters


Created on Thu July 12 21:44:23 2023

@author_ dhaneor
"""
import abc
import logging
import numpy as np
from dataclasses import dataclass
from functools import reduce
from typing import Any, NamedTuple, Optional, Sequence, Callable

from util import proj_types as tp
from .strategy import signal_generator as sg
from .strategy import exit_order_strategies as es
from .strategy.condition import ConditionResult, merge_signals_nb
from analysis.models.market_data import MarketData
from .indicators.iindicator import IIndicator
from .indicators.indicator_parameter import Parameter

logger = logging.getLogger("main.strategy_builder")


# ======================================================================================
# type alias for
CombineFuncT = Callable[[Sequence[np.ndarray]], np.ndarray]


def combine_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Combines a list of NumPy arrays by adding the values
    from each cell with the same index, excluding the first array.

    Parameters
    ----------
    arrays: Sequence[np.ndarray]
        a sequence of NumPy arrays, which must have identical shape

    Returns
    -------
    result: np.ndarray
    """
    result = np.array(arrays[0])
    for array in arrays[1:]:
        result = np.add(result, array)
    return result


# ======================================================================================
@dataclass(frozen=True)
class StrategyDefinition:
    """Complete (single or composite) strategy definition.

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this case.

    Parameters
    ----------
    strategy: str
        name of the strategy

    symbol: str
        symbol to trade

    interval: str
        interval to trade

    weight: float
        weight of the strategy, defaults to 1

    params: Dict[str, Any]
        parameters for the strategy, defaults to None

    sub_strategies: Sequence[StrategyDefinition]
        sequence of sub strategies, defaults to None

    stop_loss: Sequence[es.IStopLossStrategy]
        sequence of stop loss strategies, defaults to None

    take_profit: Sequence[es.ITakeProfitStrategy]
        sequence of take profit strategies, defaults to None
    """

    strategy: str
    symbol: str
    interval: str
    signals_definition: Optional[sg.SignalsDefinition] = None
    weight: float = 1.0
    rebalance: bool = False
    params: dict[str, Any] | None = None
    sub_strategies: Sequence[NamedTuple] | None = None
    stop_loss: Optional[Sequence[es.StopLossDefinition]] = None
    take_profit: Optional[Sequence[es.StopLossDefinition]] = None


class IStrategy(abc.ABC):
    """Interface for all strategy classes."""

    def __init__(self, name: str, params: dict | None = None) -> None:
        self.name: str = name
        self.symbol: str = ""
        self.interval: str = ""

        self._market_data: MarketData = None

        self.is_sub_strategy: bool = False
        self.weight: float = 1.0
        self.params: dict[str, Any] = params or {}

        self.sl_strategy: Optional[Sequence[es.IStopLossStrategy]] = None
        self.tp_strategy: Optional[Sequence[es.ITakeProfitStrategy]] = None

        self.sub_strategies: Sequence[IStrategy] = []

        self.definition: Optional[StrategyDefinition] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.sl_strategy}, {self.tp_strategy})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} for {self.symbol} "\
            f"({self.sl_strategy}, {self.tp_strategy})"

    @property
    def market_data(self) -> MarketData:
        return self._market_data
    
    @market_data.setter
    def market_data(self, market_data: MarketData) -> None:
        if self.is_sub_strategy:
            self.signal_generator.market_data = market_data
        else:
            for strategy in self.sub_strategies:
                strategy.market_data = market_data

    # ----------------------------------------------------------------------------------
    @abc.abstractmethod
    def speak(self, data: tp.Data) -> tp.Data:
        """Calculates signals, stop loss, take profit and positions

        This is the main entry point for getting results from the
        strategy.

        Parameters
        ----------
        data: tp.Data
            'data' must be a dictionary with the following keys:
            'o', 'h', 'l', 'c', 'v' (OHLCV data for one symbol)

        Returns
        -------
        data: tp.Data
            data with added signals, values for stop loss and
            take profit, and the (theoretical) positions
        """
        ...

    @abc.abstractmethod
    def randomize(self) -> None:
        ...


# ======================================================================================
class SubStrategy(IStrategy):
    """Base class for all strategy classes with only one strategy.

    Use the composite strategy class instead of this class, if
    you want to combine the signals (and their weight) from
    multiple of those simple strategies!
    """
    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self.signal_generator: sg.SignalGenerator
        self.is_sub_strategy = True

    def __repr__(self) -> str:

        try:
            sg_str = f'{str(self.signal_generator)}'
        except AttributeError:
            sg_str = 'None'

        sl_str = ''
        if self.sl_strategy:
            sl_str = '\n\t\t'.join([str(s) for s in self.sl_strategy])

        if not sl_str:
            sl_str = 'None'

        tp_str = ''
        if self.tp_strategy:
            tp_str = '\n\t\t'.join([str(s) for s in self.tp_strategy])

        if not tp_str:
            tp_str = 'None'

        return (
            f"{self.name} strategy for {self.symbol} "
            f'in {self.interval} interval ({self.weight=:.2f})'
            f'\n\t {sg_str}'
            f'\n\tstop loss: {sl_str}'
            f'\n\ttake profit: {tp_str}'
        )

    def __str__(self) -> str:
        return self.__repr__()

    # --------------------------------------------------------------------------
    @property
    def indicators(self) -> tuple[IIndicator, ...]:
        return self.signal_generator.indicators

    @property
    def parameters(self) -> tuple[Parameter, ...] | None:
        return self.signal_generator.paramaters

    @parameters.setter
    def parameters(self, value: tuple[Any, ...]) -> None:
        self.signal_generator.paramaters = value

    # --------------------------------------------------------------------------
    def speak(self) -> tp.Data:
        if self.weight == 1:
            return self.signal_generator.execute()
        else:
            return np.multiply(
                self.signal_generator.execute(), 
                self.weight
                )
    
    def randomize(self) -> None:
        """Randomizes the parameters of the strategy.
        """
        self.signal_generator.randomize()

    def optimize(self) -> None:
        """
        # This method should be implemented in the subclasses to optimize the strategy
        pass
        """
        raise NotImplementedError(f"optimize() not implemented for {self.name}")


class CompositeStrategy(IStrategy):
    """Base class for all composite strategies."""

    def __init__(self, symbol: str, interval: str, weight: float = 1):
        super().__init__()

        self.symbol = symbol
        self.interval = interval
        self.weight = weight
        self._combine_func: CombineFuncT

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.symbol}, {self.interval})"

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}({self.symbol}, {self.interval})"
        string += f"\n\tstop loss:   {self.sl_strategy}"
        string += f"\n\ttake profit: {self.tp_strategy}"
        string += f"\n\t{len(self.sub_strategies)} sub-strategies:"

        for name, (strategy, weight) in self.sub_strategies.items():
            string += f"\n\t\t{name}: {strategy} ({weight})"

        return string

    def speak(self) -> tp.Data:
        """Calculates signals and adds them to the data dictionary.

        Use this method to define the actual behaviour in the sub-classes!

        Parameters
        ----------
        data : tp.Data
            OHLCV data dictionary

        Returns
        -------
        tp.Data
            data with added 'signal' key/values
        """
        # get the (combined) signal from all sub-strategies. These are
        # returned as ConditionResult objects, and we extract only the
        # combined signal here.
        try:
            condition_results = tuple(sub.get_signals() for sub in self.sub_strategies)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise

        logger.debug("Got signals from %s sub-strategies" % len(condition_results))

        # add the signals column from each sub-strategy to the data dictionary
        # as_dict = {
        #     f"signal.{idx + 1}": elem for idx, elem in enumerate(condition_results)
        #     }

        # combine the signals from all sub-strategies into one
        # combined_signal = reduce(lambda x, y: np.add(x, y), condition_results)

        # add the combine signal to the data dictionary. We need to
        # construct a ConditionResult object first to get all columns
        # (for open_long, open_short, etc., plus the combined signal)
        # data.update(
        #     ConditionResult
        #     .from_combined(combined_signal)
        #     .as_dict()
        #     )

        return data.update(condition_results[0])

    def randomize(self) -> None:
        """Randomize the parameters of the strategies.
        """
        for strategy, _ in self.sub_strategies.values():
            strategy.randomize()


# ======================================================================================
def build_sub_strategy(sdef: StrategyDefinition) -> IStrategy:
    """Builds a single strategy class from a strategy definition.

    Single strategy means that it does not have sub-strategies.

    Parameters
    ----------
    sdef: StrategyDefinition
        a formalized strategy definition

    Returns
    -------
    strategy: IStrategy
        the strategy class
    """
    logger.debug("building single strategy: %s - %s", sdef.strategy, sdef.params)

    # create the strategy class from the template for single strategies
    strategy = SubStrategy(name=sdef.strategy, params=sdef.params)
    strategy.symbol = sdef.symbol
    strategy.interval = sdef.interval
    strategy.weight = sdef.weight or 1.0
    strategy.is_sub_strategy = True

    strategy.signal_generator = sg.signal_generator_factory(sdef.signals_definition)
    strategy.definition = sdef

    # add stop loss strategy if provided
    if sdef.stop_loss:
        strategy.sl_strategy = tuple(
            es.sl_strategy_factory(d) for d in sdef.stop_loss
            )

    # add take profit strategy if provided
    if sdef.take_profit:
        strategy.tp_strategy = tuple(
            es.tp_strategy_factory(d) for d in sdef.take_profit
            )

    return strategy


def build_strategy(sdef: StrategyDefinition) -> IStrategy:
    """Builds a strategy class from a strategy definition.

    Parameters
    ----------
    sdef: StrategyDefinition
        a formalized strategy definition

    Returns
    -------
    strategy: IStrategy
        the strategy class

    Raises
    ------
    ValueError
        for missing 'name' in strategy_definition

    ValueError
        for missing'symbol' in strategy_definition

    ValueError
        for missing 'interval' in strategy_definition
    """

    # validation logic
    def _validate_strategy_definition(sdef: StrategyDefinition) -> None:
        if sdef.strategy is None:
            raise ValueError(f"strategy: {sdef.strategy} is not a valid strategy")

        if sdef.symbol is None:
            raise ValueError(f"strategy: {sdef.strategy} requires a symbol")

        if sdef.interval is None:
            raise ValueError(f"strategy: {sdef.strategy} requires an interval")

    # ..........................................................................
    # build a single strategy if sub_strategies is not provided/requested
    # this simplifies definitions for single (indicator) strategies
    # and makes the strategy a little bit faster later on
    _validate_strategy_definition(sdef)

    if not sdef.sub_strategies:
        try:
            return build_sub_strategy(sdef)
        except ValueError as err:
            logger.error("failed to build (simple) strategy %s: %s", sdef.strategy, err)
            raise

    # recursively build a composite strategy if sub_strategies is provided
    # this allows sophisticated strategies to be built, that can contain
    # multiple sub_strategies, which can also be a composite strategy.
    else:
        logger.debug("building composite strategy: %s \n%s\n", sdef.strategy, sdef)

        strategy = CompositeStrategy(sdef.symbol, sdef.interval, sdef.weight)
        strategy.name = sdef.strategy

        try:
            strategy.sub_strategies = {
                sub_def.strategy: (
                    build_strategy(sub_def),
                    sub_def.weight,
                )
                for sub_def in sdef.sub_strategies
            }
        except ValueError as err:
            logger.error("failed to build %s: %s", sdef.strategy.upper, err)
            raise
        except Exception as err:
            logger.error("failed to build %s: %s", sdef.strategy.upper, err)
            raise

        # add stop loss strategy if provided
        if sdef.stop_loss:
            strategy.sl_strategy = [
                es.sl_strategy_factory(sl_def) for sl_def in sdef.stop_loss
            ]

        # add take profit strategy if provided
        if sdef.take_profit:
            strategy.tp_strategy = [
                es.tp_strategy_factory(tp_def) for tp_def in sdef.take_profit
            ]

        return strategy


async def strategy_repository() -> CompositeStrategy:
    ...