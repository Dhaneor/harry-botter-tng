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

from .util import proj_types as tp
from .strategies import signal_generator as sg
from .strategies import exit_order_strategies as es

logger = logging.getLogger("main.strategy_builder")
# logger.setLevel(logging.INFO)


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
class IStrategy(abc.ABC):
    """Interface for all strategy classes."""

    def __init__(self) -> None:
        self.name: str = ""
        self.symbol: str = ""
        self.interval: str = ""

        self.is_sub_strategy: bool = False
        self.weight: tp.Weight = 1.0
        self.params: dict[str, Any] = {}

        self.sl_strategy: Optional[Sequence[es.IStopLossStrategy]] = None
        self.tp_strategy: Optional[Sequence[es.ITakeProfitStrategy]] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.sl_strategy}, {self.tp_strategy})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} for {self.symbol} "\
            f"({self.sl_strategy}, {self.tp_strategy})"

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

    # ----------------------------------------------------------------------------------
    @abc.abstractmethod
    def _add_signals(self, data: tp.Data) -> None:
        ...

    def _add_stop_loss(self, data) -> np.ndarray:
        if self.sl_strategy:
            return self.sl_strategy.add_stop_loss(data)

        return data

    def _add_take_profit(self, data) -> np.ndarray:
        if self.tp_strategy:
            return self.tp_strategy.add_take_profit(data)

        return data

    def _add_positions(self, data: tp.Data) -> None:
        """Calculates positions and adds them to the data dictionary.

        Parameters
        ----------
        data: tp.Data
            dict with OHLCV data & signals
        """
        raise NotImplementedError(f"_add_positions() not implemented for {self.name}")

    def _add_position_sizes(self, data: tp.Data) -> None:
        """Calculates position sizes and adds them to the data dictionary.

        Parameters
        ----------
        data: tp.Data
            dict with OHLCV data & signals
        """
        raise NotImplementedError(
            f"_add_position_sizes() not implemented for {self.name}"
        )


# ======================================================================================
class SubStrategy(IStrategy):
    """Base class for all strategy classes with only one strategy.

    Use the composite strategy class instead of this class, if
    you want to combine the signals (and their weight) from
    multiple of those simple strategies!
    """
    def __init__(self, name: str, params: Optional[dict] = None):
        super().__init__()
        self.name: str = name
        self._signal_generator: sg.SignalGenerator

    def __repr__(self) -> str:

        try:
            sg_str = f'{str(self._signal_generator)}'
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
    def indicators(self) -> tuple:
        return self._signal_generator.indicators

    # --------------------------------------------------------------------------
    def speak(self, data: tp.Data) -> tp.Data:
        for key in sg.SignalGenerator.dict_keys:
            if key not in data:
                data[key] = np.zeros(data['open'].shape)

        return self._add_take_profit(
            self._add_stop_loss(
                self._add_signals(data)
            )
        )

    # --------------------------------------------------------------------------
    def _add_signals(self, data: tp.Data) -> tp.Data:
        """Calculates signals and adds them to the data dictionary.

        Use this method to define the actual behaviour in the sub-classes!

        :param data: dict with OHLCV data for one symbol
        :type data: _type_
        :return: _description_
        :rtype: _type_
        """
        return self._signal_generator.execute(data, self.weight)

    def _add_stop_loss(self, data: tp.Data) -> tp.Data:
        """Calculates stop loss and adds it to the data dictionary.

        Parameters
        ----------
        data: tp.Data
            dict with OHLCV data

        Returns
        -------
        tp.Data
            the dict with added 'stop loss' series
        """
        return data
        return super()._add_stop_loss(data)

    def _add_take_profit(self, data: tp.Data) -> tp.Data:
        """Calculates take profit and adds it to the data dictionary.

        Parameters
        ----------
        data: tp.Data
            dict with OHLCV data

        Returns
        -------
        tp.Data
            the dict with added 'take profit' series
        """
        return data
        return super()._add_take_profit(data)


class CompositeStrategy(IStrategy):
    """Base class for all composite strategies."""

    def __init__(self, symbol: str, interval: str, weight: float = 1):
        super().__init__()

        self.symbol = symbol
        self.interval = interval
        self.weight = weight
        self._combine_func: CombineFuncT

        self.sub_strategies: dict[str, tuple[IStrategy, tp.Weight]] = {}

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

    def speak(self, data: tp.Data) -> tp.Data:
        self._add_signals(data)

    def optimize(self) -> None:
        """Optimize the strategy by testing param combinations.

        Raises
        ------
        NotImplementedError
            for now :P
        """
        raise NotImplementedError("optimize is waiting to be implemented")

    def backtest(self, data: tp.Data) -> None:
        """Backtest the strategy.

        Parameters
        ----------
        data: tp.Data
            OHLCV data dictionary

        Raises
        ------
        NotImplementedError
            for now :P
        """
        raise NotImplementedError("backtest is waiting to be implemented")

    # ..................................................................................
    def _add_signals(self, data: tp.Data) -> tp.Data:
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
        for key in sg.SignalGenerator.dict_keys:
            if key not in data:
                logger.debug("creating key '%s' in data", key)
                data[key] = np.zeros(data['open'].shape)

        [strat.speak(data) for strat, _ in self.sub_strategies.values()]

        # data['signal'] = self._combine_signals(signals)

        return data

    def _combine_signals(self, signals: Sequence[np.ndarray]) -> np.ndarray:
        """Combines the signals from multiple sub-strategies.

        Parameters
        ----------
        signals: Signals -> Dict[str, Tuple[np.ndarray, float]]
            signals from multiple sub-strategies
        Returns
        -------
        np.ndarray
            combined signals
        """
        return combine_arrays(signals)

    def _add_stop_loss(self, data: tp.Data) -> tp.Data:
        """Calculates stop loss and adds it to the data dictionary.

        Parameters
        ----------
        data: dict with OHLCV data for one symbol

        Returns
        -------
        data: tp.Data
            the dict with added 'stop loss' series
        """
        return super()._add_stop_loss(data)

    def _add_take_profit(self, data: tp.Data) -> tp.Data:
        """Calculates take profit and adds it to the data dictionary.

        Parameters
        ----------
        data: dict with OHLCV data for one symbol

        Returns
        -------
        data: tp.Data
            the dict with added 'take profit' series
        """
        return super()._add_take_profit(data)


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
    params: Optional[dict[str, str | float | int | bool]] = None
    sub_strategies: Optional[Sequence[NamedTuple]] = None
    stop_loss: Optional[Sequence[es.StopLossDefinition]] = None
    take_profit: Optional[Sequence[es.StopLossDefinition]] = None


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

    Raises
    ------
    ValueError
        if sdef.strategy is not a valid strategy
    """
    logger.debug("building single strategy: %s - %s", sdef.strategy, sdef.params)

    # if not isinstance(c, cnd.ConditionDefinition):
    #     raise ValueError(f'invalid condition definition: {c}')

    # create the strategy class from the template for single strategies
    strategy = SubStrategy(name=sdef.strategy, params=sdef.params)
    strategy.symbol = sdef.symbol
    strategy.interval = sdef.interval
    strategy.weight = sdef.weight
    strategy.is_sub_strategy = True

    strategy._signal_generator = sg.factory(sdef.signals_definition)

    # add stop loss strategy if provided
    if sdef.stop_loss:
        strategy.sl_strategy = [es.sl_strategy_factory(d) for d in sdef.stop_loss]

    # add take profit strategy if provided
    if sdef.take_profit:
        strategy.tp_strategy = [es.tp_strategy_factory(d) for d in sdef.take_profit]

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
    # this simpllfies definitions for single (indicator) strategies
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
        logger.debug("building composite strategy: %s - %s", sdef.strategy, sdef)

        strategy = CompositeStrategy(sdef.symbol, sdef.interval, sdef.weight)

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
