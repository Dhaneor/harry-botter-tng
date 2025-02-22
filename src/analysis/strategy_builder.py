#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides tools to build executable strategies dynamically.

functions:
    build_strategy(sdef: StrategyDefinition) -> IStrategy:
        main entry point for the module, should be the only function
        that a user/client needs

    build_strategy_single(sdef: StrategyDefinition) -> Strategy:
        Builds a single strategy.

        This is a helper function for build_strategy(). Because
        there are single and composite strategies, this function
        is specifically used for building single strategies.
        Single strategies are the elements for the sub_strategies
        attribute of a composite strategy. From a user perspective,
        there are only composite strategies, even if they contain
        only one sub-strategy.


classes:
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
from typing import Any, NamedTuple, Optional, Sequence

from .strategy import signal_generator as sg
from .strategy import exit_order_strategies as es
from analysis.models.market_data import MarketData
from analysis.models.signals import SignalStore
from analysis.leverage import LeverageCalculator
from analysis.backtest.backtest import Config, BackTestCore
from analysis.backtest.bt_result import BackTestResult

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
        self.risk_level: int = 1

        self.sub_strategies: Sequence[IStrategy] = []
        self.definition: Optional[StrategyDefinition] = None
        self.is_sub_strategy: bool = True

        self.add_signals: bool = True
        self.normalize_signals: bool = False

        self._market_data: MarketData = None
        self._leverage_calculator = None


    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def market_data(self) -> MarketData:
        if self._market_data is None:
            raise ValueError("market_data not initialized")
        return self._market_data
    
    @market_data.setter
    def market_data(self, market_data: MarketData) -> None:
        logger.info(
            "%s %s has been assigned market data",
            "Sub-Strategy" if self.is_sub_strategy else "Composite Strategy",
            self.name
            )
        if not self.sub_strategies:
            self.signal_generator.market_data = market_data
            self._leverage_calculator = LeverageCalculator(market_data=market_data)
        else:
            for strategy in self.sub_strategies:
                strategy.market_data = market_data

    # ----------------------------------------------------------------------------------
    @abc.abstractmethod
    def run(self):
        ...
    
    def speak(self) -> SignalStore:
        """Calculates signals for the current market data.
        
        Returns:
        --------
        SignalStore
            A SignalStore object containing the calculated signals. If 
            multiple parameter combinations are used, the signals will 
            be summed up, but not normalized.
        """
        signals = self._get_raw_signals()

        if self.add_signals and self.normalize_signals:
            return SignalStore(signals).summed(normalized=True)
        elif self.add_signals and not self.normaalize_signals:
            return SignalStore(signals).summed(normalized=False)
        elif not self.add_signals and not self.normalize_signals:
            return signals
        else:
            raise ValueError(
                "Can not normalize signals when add_signals is set to False"
                )

    @abc.abstractmethod
    def randomize(self) -> None:
        ...

    @abc.abstractmethod
    def plot(self, symbol: str):
        ...

    @abc.abstractmethod
    def _get_raw_signals(self) -> SignalStore:
        ...


# ======================================================================================
class Strategy(IStrategy):
    """Class for simple strategies.

    Use the composite strategy class instead of this class, if
    you want to combine the signals (and their weight) from
    multiple of those simple strategies!
    """
    def __init__(self, name: str, weight: float = 1) -> None:
        self.signal_generator: sg.SignalGenerator = None
        super().__init__(name, weight)
        
        self.config: Config = Config(10_000)
        self.backtest = None

    def __repr__(self) -> str:
        return (
            f"{self.name} strategy for {self.symbol} ({self.weight=:.2f})"
            f"\n\t {str(self.signal_generator)}"
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
        if attr in (
            'name', 'weight', 'signal_generator', '_market_data', 
            'sub_strategies', 'definition', 'is_sub_strategy'
        ):
            super().__setattr__(attr, value)
        elif self.signal_generator is not None:
            setattr(self.signal_generator, attr, value)
        else:
            super().__setattr__(attr, value)
        
    # ----------------------------------------------------------------------------------
    def run(self) -> None:
        signals = self._get_raw_signals()

        bt = BackTestCore(
            market_data=self.market_data.mds,
            leverage=self._get_leverage(),
            signals=signals,
            config=self.config
        )

        positions, portfolio = bt.run()

        return BackTestResult(
            positions=positions, 
            portfolio=portfolio,
            marke_data=self.market_data,
            signals=signals
        )

    def speak(self) -> SignalStore:
        return super().speak()

    def randomize(self) -> None:
        """Randomizes the parameters of the strategy."""
        self.signal_generator.randomize()

    def plot(self, symbol: str):
        raise NotImplementedError(f"plot() not implemented for {self.name}")

    def optimize(self) -> None:
        """
        # This method should be implemented in the subclasses to optimize the strategy
        pass
        """
        raise NotImplementedError(f"optimize() not implemented for {self.name}")

    def _get_raw_signals(self) -> np.ndarray:
        return self.signal_generator.execute(compact=True)

    def _get_leverage(self) -> np.ndarray:
        return self._leverage_calculator.leverage(self.risk_level)


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

    def speak(self) -> SignalStore:
        """Collects and normalizes signals from all sub-strategies.

        Returns
        -------
        SignalStore
            SignalStore instances can be added, so we can just use the 
            sum() function. The normalize() function is applied to the 
            aggregated signals.            
        """
        return super().speak()

    def randomize(self) -> None:
        """Randomize the parameters of the strategies.
        """
        for strategy in self.sub_strategies:
            strategy.randomize()

    def optimize(self) -> None:
        for strategy in self.sub_strategies:
            strategy.optimize()

    def plot(self) -> None:
        pass

    def _get_raw_signals(self) -> SignalStore:
        return sum((sub.speak() for sub in self.sub_strategies)).normalized()
    

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
    strategy = Strategy(name=sdef.strategy, weight=sdef.weight)
    strategy.symbol = sdef.symbol
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