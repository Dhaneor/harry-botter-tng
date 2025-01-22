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
from typing import Any, NamedTuple, Optional, Sequence, Callable

from util import proj_types as tp
from .strategy import signal_generator as sg
from .strategy import exit_order_strategies as es
from analysis.models.market_data import MarketData

logger = logging.getLogger("main.strategy_builder")


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

    def __init__(self, name: str, weight: float) -> None:
        self.name: str = name
        self.weight: float = weight

        self._market_data: MarketData = None
        self.sub_strategies: Sequence[IStrategy] = []
        self.definition: Optional[StrategyDefinition] = None
        self.is_sub_strategy: bool = True

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def market_data(self) -> MarketData:
        return self._market_data
    
    @market_data.setter
    def market_data(self, market_data: MarketData) -> None:
        if not self.sub_strategies:
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
    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.signal_generator: sg.SignalGenerator = None
        super().__init__(name, weight)

    def __repr__(self) -> str:

        try:
            sg_str = f'{str(self.signal_generator)}'
        except AttributeError:
            sg_str = 'None'

        return (
            f"{self.name} strategy for {self.symbol} "
            f'in {self.interval} interval ({self.weight=:.2f})'
            f'\n\t {sg_str}'
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __getattr__(self, attr) -> Any:
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif self.signal_generator is not None:
            return getattr(self.signal_generator, attr)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{attr}'"
            )

    def __setattr__(self, attr, value) -> None:
        if attr in ('name', 'weight', 'signal_generator', '_market_data', 'sub_strategies', 'definition', 'is_sub_strategy'):
            super().__setattr__(attr, value)
        elif self.signal_generator is not None:
            setattr(self.signal_generator, attr, value)
        else:
            super().__setattr__(attr, value)
        
    # ----------------------------------------------------------------------------------
    def speak(self) -> tp.Data:
        if self.weight == 1.0:
            return self.signal_generator.execute(compact=True)
        else:
            return np.multiply(
                self.signal_generator.execute(compact=True), 
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

    def __init__(self, name: str = "Unnamed Strategy", weight: float = 1):
        super().__init__(name, weight)
        self.is_sub_strategy = False

    def __repr__(self) -> str:
        string = f"{self.name})"
        string += f"\n\t{len(self.sub_strategies)} sub-strategies:"

        for sub in self.sub_strategies:
            string += f"\n\t\t{sub.name} ({sub.weight})"

        return string

    def __str__(self) -> str:
        return self.__repr__()

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
        return np.sum(sub.get_signals() for sub in self.sub_strategies)


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
    strategy = SubStrategy(name=sdef.strategy)
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

        strategy = CompositeStrategy(sdef.strategy, sdef.weight)

        try:
            strategy.sub_strategies = [
                build_strategy(sub_def) for sub_def in sdef.sub_strategies
            ]
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