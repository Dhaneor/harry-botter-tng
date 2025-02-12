#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the interface/abstract base class for all strategy classes,
and also a generic single/simple strategy class as well as a generic
composite strategy class.

These are the building blocks for all concrete strategy implementations,
which are requested from and built by the strategy builder, based on
standardized strategy definitions.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ...util import proj_types as tp
from .signal_generator import SignalGenerator
from .exit_order_strategies import IStopLossStrategy, ITakeProfitStrategy

logger = logging.getLogger('main.i_strategy')
logger.setLevel(logging.DEBUG)

# type alias for
CombineFuncT = Callable[[Sequence[np.ndarray]], np.ndarray]


def combine_arrays(arrays: Sequence[np.ndarray]):
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


# =============================================================================
class IStrategy(ABC):
    """Interface for all strategy classes."""

    def __init__(self) -> None:
        self.NAME: str = ""
        self.symbol: str = ""
        self.interval: str = ""

        self.weight: tp.Weight = 1.0
        self.params: dict[str, Any] = {}

        self.sl_strategy: Optional[Sequence[IStopLossStrategy]] = None
        self.tp_strategy: Optional[Sequence[ITakeProfitStrategy]] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.sl_strategy}, {self.tp_strategy})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} for {self.symbol} "\
            f"({self.sl_strategy}, {self.tp_strategy})"

    @abstractmethod
    async def speak(self, data: tp.Data) -> tp.Data:
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

    @abstractmethod
    async def _add_signals(self, data: tp.Data) -> tp.Data:
        ...

    async def _add_stop_loss(self, data):
        if self.sl_strategy:
            return await self.sl_strategy.add_stop_loss(data)
        else:
            return data

    async def _add_take_profit(self, data):
        if self.tp_strategy:
            return await self.tp_strategy.add_take_profit(data)
        else:
            return data


class BaseStrategySingle(IStrategy):
    """Base class for all strategy classes with only one strategy.

    Use the composite strategy class instead of this class, if
    you want to combine the signals (and their weight) from
    multiple of those simple strategies!
    """
    def __init__(self, name: str, params: Optional[dict] = None):
        self.NAME: str

        super().__init__()

        self._signal_generator: SignalGenerator
        self._VALID_PARAMS: set = set()
        self._default_params: dict[str, Any] = {}

        if params:
            self.default_params = params

    def __repr__(self) -> str:
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
            f"{self.NAME} strategy for {self.symbol} "
            f'in {self.interval} interval ({self.weight=:.2f})\n'
            f'\tdefault params: {self._default_params}'
            f'\n\tstop loss: {sl_str}'
            f'\n\ttake profit: {tp_str}'
        )

    def __str__(self) -> str:
        return self.__repr__()

    # --------------------------------------------------------------------------
    @property
    def VALID_PARAMS(self) -> set:
        """Returns the set of valid parameters for the strategy class.

        These should not and can not be changed by the user/client!

        Returns
        -------
        set
            all valid parameters for this strategy
        """
        return self._VALID_PARAMS

    @property
    def default_params(self) -> dict[str, Any]:
        """Returns the default parameters for the strategy.

        Returns
        -------
        Dict[str, Any]
            all default parameters
        """
        return self._default_params

    @default_params.setter
    def default_params(self, params: dict[str, Any]) -> None:
        """Sets the default parameters for the strategy.

        If the parameter is not in self.VALID_PARAMS, it will be ignored
        and a warning will be logged.

        :param params: the parameters to be changed
        :type params: Dict[str, Any]
        """
        [
            self._default_params.update({k: v})
            if k in self.VALID_PARAMS
            else logger.warning(
                f"parameter '{k}' is not a valid parameter "
                f"for {self.__class__.__name__}"
            )
            for k, v in params.items()
        ]

    # --------------------------------------------------------------------------
    def speak(self, data: tp.Data) -> tp.Data:
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
        # the base strategy adds random signals (just used for testing)
        data['signal'] = np.random.choice({0, 1, -1}, size=len(data['o']))
        return data

    def _add_stop_loss(self, data: tp.Data) -> tp.Data:
        """Calculates stop loss and adds it to the data dictionary.

        :param data: dict with OHLCV data for one symbol
        :type data: _type_
        :return: _description_
        :rtype: _type_
        """
        return super()._add_stop_loss(data)

    def _add_take_profit(self, data: tp.Data) -> tp.Data:
        """Calculates take profit and adds it to the data dictionary.

        :param data: dict with OHLCV data for one symbol
        :type data: _type_
        :return: _description_
        :rtype: _type_
        """
        return super()._add_take_profit(data)


class BaseStrategyComposite(IStrategy):
    """Base class for all composite strategies."""

    def __init__(self, symbol: str, interval: str, weight: float = 1):
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
        pass

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
        signals = {}
        for name, (strategy, weight) in self.sub_strategies.items():
            signals[name] = strategy.speak(data), weight

        data['signal'] = self._combine_signals(signals)
        return data

    def _combine_signals(self, signals: Sequence[np.ndarray]) -> np.ndarray:
        """Combines the signals from multiple sub-strategies.

        :param signals: signals from multiple sub-strategies
        :type signals: Signals -> Dict[str, Tuple[np.ndarray, float]]
        :return: combined signals
        :rtype: np.ndarray
        """
        return combine_arrays(signals)

    async def _add_stop_loss(self, data: tp.Data) -> tp.Data:
        """Calculates stop loss and adds it to the data dictionary.

        :param data: dict with OHLCV data for one symbol
        :type data: _type_
        :return: the dict with added 'stop loss' series
        :rtype: Dict[str, np.ndArray]
        """
        return super()._add_stop_loss(data)

    async def _add_take_profit(self, data: tp.Data) -> tp.Data:
        """Calculates take profit and adds it to the data dictionary.

        :param data: dict with OHLCV data for one symbol
        :type data: _type_
        :return: the dict with added 'take profit' series
        :rtype: Dict[str, np.ndArray]
        """
        return super()._add_take_profit(data)
