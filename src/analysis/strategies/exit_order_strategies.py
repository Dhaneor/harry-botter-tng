#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides diffetent exit order strategies (stop loss/take profit)
and the interface for all of them.

Use as interface:
- IStopLossOrderStrategy for stop loss order strategies
- ITakeProfitOrderStrategy for take profit order strategies

Instances should only be created via the factory function(s):
- sl_strategy_factory(sl_def: StopLossDefinition)
- tp_strategy_factory(tp_def: TakeProfitDefinition)

Use get_valid_strategies() to see all options available!

Created on Fri Nov 25 20:25:20 2022

@author dhaneor
"""
import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, TypeAlias, NamedTuple

from ..indicators.indicators_fast_nb import atr
# from ..indicators import indicator_parameter as param

logger = logging.getLogger("main.exit_order_strategies")

# initialize (compile) atr function. this is done once here,
# so on the first invocation teh Numba compiled function
# is available instantly.
_ = atr(
    np.zeros(100, dtype=np.float64),
    np.zeros(100, dtype=np.float64),
    np.zeros(100, dtype=np.float64),
    14,
)

# type hint types
price = float
fraction = float
LongStopLossPrices: TypeAlias = np.ndarray
ShortStopLossPrices: TypeAlias = np.ndarray


# =============================================================================

class StopLossDefinition(NamedTuple):
    """A formal definition of a stop loss strategy.

    Attributes:
    ----------
    strategy: str
        the name of the stop loss strategy to be used

    params: Union[dict, None]
        the parameters of the stop loss strategy, default: None
    """
    strategy: str
    params: Union[dict, None] = None


class TakeProfitDefinition(NamedTuple):
    """A formal definition of a take profit strategy.

    Attributes:
    ----------
    strategy: str
        the name of the take profit strategy to be used

    params: Union[dict, None]
        the parameters of the take profit strategy, default: None
    """

    strategy: str
    params: Union[dict, None] = None


# =============================================================================
class IExitOrderStrategy(ABC):
    """Interface class for all exit order strategies."""

    NAME: str

    def __init__(self, params: Optional[dict] = None) -> None:
        """Initializes an exit order strategy.

        Parameters:
        ----------
        params: Optional[dict]
            the parameters of the strategy, default: None
        """
        self.fractions: int = 1
        self.is_trailing: bool = False
        if params:
            self.set_params(params)

    def __repr__(self):
        print_params = {
            k: v for k, v in self.__dict__.items() if not k.startswith("logger")
        }

        return f"{self.__class__.__name__} - {print_params}"

    def set_params(self, params: dict):
        """Sets the parameters of the strategy.

        Is called after the strategy has been instantiated. Can also be
        used to change parameters on the fly. If the dictionary contains
        parameters that are not recognized by the strategy, they are
        ignored and a warning message is logged.

        Parameters:
        ----------
        params: dict
            the parameters of the strategy
        """
        exclude_params = ("strategy",)

        [
            (
                setattr(self, k, v)
                if hasattr(self, k) and k not in exclude_params
                else logger.warning(f"invalid parameter <{k}> for {self}")
            )
            for k, v in params.items()
        ]

    @abstractmethod
    def get_trigger_prices_np(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


# =============================================================================
#                               STOP LOSS STRATEGIES                          #
# =============================================================================
class IStopLossStrategy(IExitOrderStrategy):
    """Abstract base class for all stop loss strategies."""

    def __init__(self, params):
        super().__init__(params)

    @abstractmethod
    def add_stop_loss_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_stop_loss_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:
        """Calculates stop loss price(s) and their fractions.

        Usually the method returns a tuple within a tuple with a price
        and fraction=1.
        For concrete implementations that work with multiple stop losses
        the fractions will add up to one and 'fraction' describes the
        fraction of a position that will be sold/bought when the price
        reaches this price level.

        :param df: a DataFrame with at least a 'close' column
        :type df: pd.DataFrame
        :param long_or_short: values for 'LONG' or 'SHORT' position
        :type long_or_short: bool
        :return: stop price(s) and fraction(s), e.g.: ((100, 0.7), (90, 0.3))
        :rtype: Tuple[Tuple[float, float]]
        """
        self.add_stop_loss_prices_to_df(df)

        str_ = "sl_long" if long_or_short.lower() == "long" else "sl_short"
        fraction = 1 / self.fractions

        return tuple(
            tuple((df[col].iat[-1], fraction)) for col in df.columns if str_ in col
        )


# =============================================================================
class PercentStopLossStrategy(IStopLossStrategy):
    """Calculates stop loss prices based on a percent value."""

    NAME = "percent"

    def __init__(self, params: Union[dict, None] = None):
        """Calculates stop loss prices based on a percent value.

        :param params: strategy parameters (see below), defaults to None
        :type params: Union[dict, None], optional

        possible parameters for this strategy:

        percent (float): distance stop-loss to last close price

        is_trailing (bool): trailing stop-loss yes/no, defaults to: False

        fractions (int): if this is > 1, the returned values will be
        tuples with multiple and equally spaced prices and the fraction
        to sell at this level. For instance:
        ((<stop price 1>, 0.5), (<stop price 2>, 0.5))
        Otherwise the returned value will be like this: ((<stop price>, 1))
        defaults to: 1
        """
        self.percent: float = 50
        super().__init__(params)
        if params:
            self.set_params(params)

    def get_trigger_prices_np(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        pass

    def get_stop_loss_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:

        return super().get_stop_loss_prices(df=df, long_or_short=long_or_short)

    def add_stop_loss_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fraction = 1 / self.fractions

        for i in range(self.fractions):
            diff = df["close"] * self.percent / 100 * ((i + 1) * fraction)
            df[f"sl.long.{i+1}"] = df["close"] - diff
            df[f"sl.short.{i+1}"] = df["close"] + diff
        return df


class AtrStopLossStrategy(IStopLossStrategy):
    """Calculates stop loss prices based on ATR."""

    NAME = "atr"

    def __init__(self, params: Union[dict, None] = None):
        """ATR based stop-loss strategy

        Parameters:
        -----------
        params: Union[dict, None]
            parameters for the stop-loss strategy

            possible parameters:

                - atr_lookback: lookback for ATR calculation
                - atr_factor: multiply ATR by this factor
                - is_trailing (bool): trailing stop-loss yes/no, defaults to: False
        """
        self.atr_lookback: int = 14
        self.atr_factor: float = 3
        super().__init__(params)

    def add_stop_loss(self, data: dict[np.ndarray]) -> dict[np.ndarray]:
        """Calculates stop loss prices and adds them to the data dictionary."""
        long_stop_loss_prices, short_stop_loss_prices = self.get_trigger_prices_np(
            open_=data["open"], high=data["high"], low=data["low"], close=data["close"]
        )
        data["sl_long"] = long_stop_loss_prices
        data["sl_short"] = short_stop_loss_prices
        return data

    def get_trigger_prices_np(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[LongStopLossPrices, ShortStopLossPrices]:
        """The fastest way to calculate stop loss prices.

        Accepts only Numpy arrays and does not work with Pandas objects.

        Parameters
        ----------
        open_ : np.ndarray
           open prices
        high : np.ndarray
            high prices
        low : np.ndarray
            low prices
        close : np.ndarray
            close prices

        Returns
        -------
        Tuple[LongStopLossPrices, ShortStopLossPrices]
            two Numpy arrays, one for long and one for short stop loss prices
        """
        diff = atr(open_, high, low, self.atr_lookback) * self.atr_factor
        return np.subtract(close, diff), np.add(close, diff)

    def get_stop_loss_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:

        return super().get_stop_loss_prices(df=df, long_or_short=long_or_short)

    def add_stop_loss_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fraction = 1 / self.fractions

        atr_series = atr(
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            self.atr_lookback,
        )

        for i in range(self.fractions):
            diff = atr_series * self.atr_factor * ((i + 1) * fraction)
            df[f"sl_long_{i+1}"] = df["close"] - diff
            df[f"sl_short_{i+1}"] = df["close"] + diff
        return df


# =============================================================================
#                               TAKE PROFIT STRATEGIES                        #
# =============================================================================
class ITakeProfitStrategy(IExitOrderStrategy):
    """Abstract base class for all take profit strategies."""

    def __init__(self, params: Union[dict, None] = None):
        super().__init__(params)

    @abstractmethod
    def add_take_profit_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_take_profit_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:
        """Calculates stop loss price(s) and their fractions.

        Usually the method returns a tuple within a tuple with a price
        and fraction=1.
        For concrete implementations that work with multiple stop losses
        the fractions will add up to one and 'fraction' describes the
        fraction of a position that will be sold/bought when the price
        reaches this price level.

        :param df: a DataFrame with at least a 'close' column
        :type df: pd.DataFrame
        :param long_or_short: values for 'LONG' or 'SHORT' position
        :type long_or_short: bool
        :return: take profit price(s) and fraction(s),
        e.g.: ((100, 0.7), (90, 0.3))
        :rtype: Tuple[Tuple[float, float]]
        """
        self.add_take_profit_prices_to_df(df)

        str_ = "tp.long" if long_or_short.lower() == "long" else "tp.short"
        fraction = 1 / self.fractions

        return tuple(
            tuple((df[col].iat[-1], fraction)) for col in df.columns if str_ in col
        )


# =============================================================================
class PercentTakeProfitStrategy(ITakeProfitStrategy):
    """Calculates take profit price(s) based on a percent value."""

    NAME = "percent"

    def __init__(self, params: Union[dict, None] = None):
        """Calculates stop loss prices based on a percent value.

        :param params: parameter dict, here only: 'percent'
        :type params: Union[dict, None], optional
        """
        self.percent: float = 50
        super().__init__(params)

    def get_trigger_prices_np(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        pass

    def get_take_profit_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:

        return super().get_take_profit_prices(df=df, long_or_short=long_or_short)

    def add_take_profit_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fraction = 1 / self.fractions

        for i in range(self.fractions):
            diff = df["close"] * self.percent / 100 * ((i + 1) * fraction)
            df[f"tp.long.{i+1}"] = df["close"] - diff
            df[f"tp.short.{i+1}"] = df["close"] + diff
        return df


class AtrTakeProfitStrategy(ITakeProfitStrategy):
    """Calculates take profit price(s) based on ATR."""

    NAME = "atr"

    def __init__(self, params: Union[dict, None] = None):
        self.atr_lookback: int = 14
        self.atr_factor: float = 3
        super().__init__(params)

    def get_trigger_prices_np(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        pass

    def get_take_profit_prices(
        self, df: pd.DataFrame, long_or_short: str
    ) -> Tuple[Tuple[price, fraction]]:

        return super().get_take_profit_prices(df=df, long_or_short=long_or_short)

    def add_take_profit_prices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fraction = 1 / self.fractions

        if "atr" not in df.columns:
            high_, low_, close_ = df["high"], df["low"], df["close"]

            # .....................................................................
            high_low = high_ - low_
            low_close = abs(low_ - close_.shift(1))
            high_close = abs(high_ - close_.shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)

            true_range = np.max(ranges, axis=1)
            atr = true_range.ewm(span=self.atr_lookback, adjust=False).mean()
        else:
            atr = df["atr"]

        for i in range(self.fractions):
            diff = atr * self.atr_factor * ((i + 1) * fraction)
            df[f"tp.short.{i+1}"] = df["close"] - diff
            df[f"tp.long.{i+1}"] = df["close"] + diff
        return df


# valid stop loss and take profit strategies.
#
# all implementations of the abstract interface must be included
# here! otherwise the factory functions will raise an error,
# when trying to create them.
valid_sl_strategies = {"pct": PercentStopLossStrategy, "atr": AtrStopLossStrategy}

valid_tp_strategies = {"pct": PercentTakeProfitStrategy, "atr": AtrTakeProfitStrategy}


def get_valid_strategies():
    """Gets the valid stop loss & take profit strategies.

    Returns
    -------
    strats: Dict[str, dict]
        name, description for all valid strategies
    """
    return {
        "stop loss:": {k: v.__doc__ for k, v in valid_sl_strategies.items()},
        "take profit": {k: v.__doc__ for k, v in valid_tp_strategies.items()},
    }


# =============================================================================
def sl_strategy_factory(sl_def: StopLossDefinition) -> IStopLossStrategy:
    """Factory function for stop loss strategies.

    Parameters:
    ----------
    sl_def : StopLossDefinition
        a definition of a stop loss strategy.

    Returns:
    -------
    IStopLossStrategy
        a concrete implementation of IStopLossStrategy

    Raises
    ------
    ValueError
        If the 'strategy' parameter in the StopLossDefinition
        is not a valid stop loss strategy.
    """
    if sl_def.strategy not in valid_sl_strategies:
        raise ValueError(f"{sl_def.strategy} is not a valid strategy")

    return valid_sl_strategies[sl_def.strategy](sl_def.params)


def tp_strategy_factory(tp_def: TakeProfitDefinition) -> ITakeProfitStrategy:
    """Factory function for take profit strategies.

    Parameters
    ----------
    tp_def : TakeProfitDefinition
        a definition of a take profit strategy.

    Returns
    -------
    ITakeProfitStrategy
        a concrete implementation of ITakeProfitStrategy

    Raises
    ------
    ValueError
        If the 'strategy' parameter in the TakeProfitDefinition
        is not a valid take profit strategy.
    """
    if tp_def.strategy not in valid_tp_strategies:
        raise ValueError(f"{tp_def.strategy} is not a valid strategy")

    return valid_tp_strategies[tp_def.strategy](tp_def.params)
