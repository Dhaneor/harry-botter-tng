#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import numpy as np
import bottleneck as bn
import logging
from numba import int8, float32
from numba.experimental import jitclass
from sys import getsizeof

from analysis.models.market_data import MarketData, MarketDataStoreJIT as MarketDataStore
from analysis.diversification_multiplier import Multiplier

logger = logging.getLogger("main.leverage")
logger.setLevel(logging.DEBUG)

MAX_CACHE_SIZE = 100_000_000

TRADING_DAYS_PER_YEAR = 365
MILLISECONDS_PER_DAY = 1000 * 60 * 60 * 24
MILLISECONDS_PER_YEAR = TRADING_DAYS_PER_YEAR * MILLISECONDS_PER_DAY

INTERVAL_IN_MS = {
    "1m": 60 * 1000,
    "3m": 180 * 1000,
    "5m": 300 * 1000,
    "15m": 900 * 1000,
    "30m": 1800 * 1000,
    "1h": 3600 * 1000,
    "2h": 7200 * 1000,
    "4h": 14400 * 1000,
    "6h": 21600 * 1000,
    "12h": 43200 * 1000,
    "1d": 86400 * 1000,
    "3d": 259200 * 1000,
    "1w": 604800 * 1000,
    "1M": MILLISECONDS_PER_YEAR,
}


class LeverageCalculator:
    """
    A class to calculate maximum leverage based on market data and risk level.

    The following dictionary maps risk levels to their corresponding
    target risk:
    • for 1-10 - this is the targeted annualized portfolio volatility,
    based on the volatility of the daily returns
    • for 11-12 this is the risk per trade, based on the ATR
    """

    RISK_LEVELS = {
        1: 0.12,
        2: 0.18,
        3: 0.24,
        4: 0.30,
        5: 0.36,
        6: 0.42,
        7: 0.48,
        8: 0.54,
        9: 0.60,
        10: 0.01,
        11: 0.011,
        12: 0.012,
    }

    def __init__(
        self,
        market_data: MarketData,
        risk_level: int = 1,
        max_leverage: float = 1.0,
        smoothing: int = 1,
        atr_window: int = 21,
    ):
        """
        Initialize the LeverageCalculator with market data and configuration parameters.

        This constructor sets up the LeverageCalculator with the provided market data
        and configuration settings. It also pre-populates the leverage cache based on
        the size of the market data.

        Parameters:
        -----------
        market_data : MarketData
            The market data object containing price and volume information.
        risk_level : int, optional
            The risk level for leverage calculations (default is 1).
        max_leverage : float, optional
            The maximum allowed leverage (default is 1.0).
        smoothing : int, optional
            The smoothing factor for calculations (default is 1).
        atr_window : int, optional
            The window size for Average True Range calculations (default is 21).

        Returns:
        --------
        None
        """
        self.market_data = market_data

        self.risk_level = risk_level
        self.max_leverage = max_leverage

        self.smoothing = smoothing
        self.atr_window = atr_window

        self.interval = market_data.interval
        self.interval_in_ms = market_data.interval_in_ms

        self.dmc = Multiplier()

        self._cache = {}

        # The ATR calculation has already been done in market_data
        # with a window size of 21. We only need to recalculate it
        # for a different window size.
        if atr_window != 21:
            self.market_data.compute_atr(atr_window)

        # pre-populate the cache ...
        if len(market_data) < 5_000:
            # ...for all risk levels for smaller datasets
            for risk_level in self.RISK_LEVELS:
                self.leverage(risk_level)
        else:
            # ...for the current risk level for larger datasets
            self.leverage(self.risk_level)

    def leverage(self, risk_level: int = None) -> np.ndarray:
        """Calculates the maximum leverage based on 'close' prices.

        Parameters
        ----------
        risk_level
            the risk level, default is 1

        Returns
        -------
        np.ndarray
            the maximum/recommended leverage

        Raises
        ------
        ValueError
            if the risk level is not valid
        """
        risk_level = risk_level or self.risk_level

        if self._cache.get(risk_level, None) is None:
            if risk_level == 0:
                lv = np.full_like(
                    self.market_data.close, self.max_leverage, dtype=np.float32
                    )
            elif 1 <= risk_level <= 10:
                lv = self._conservative_sizing(self.RISK_LEVELS[risk_level])
            elif 11 <= risk_level <= 12:
                lv = self._aggressive_sizing(self.RISK_LEVELS[risk_level])
            else:
                raise ValueError(f"Invalid risk level: {risk_level}")

            if getsizeof(self._cache) > MAX_CACHE_SIZE:
                self._cache.clear()
                logger.info("Cache cleared due to high memory usage.")

            # apply the diversification multiplier if there are multiple assets
            if self.market_data.close.shape[1] > 1:
                lv = np.multiply(
                    lv, 
                    self.dmc.get_multiplier(self.market_data.mds.close)
                )

            # Apply the maximum allowed leverage
            lv = np.minimum(lv, self.max_leverage) 
            lv = np.asarray(lv, dtype=np.float32)

            self._cache[risk_level] = lv

        return self._cache[risk_level]

    def _conservative_sizing(self, target_risk_annual: float) -> np.ndarray:
        """Calculates the maximum leverage based on 'close' prices.

        This is the method described by Robert Carver in 'Leveraged
        Trading'. It is very conservative, at least when the account
        level risk target is also calculated by the method(s) described
        in the book (he proposes to target 12% annualized volatility
        for the portfolio).

        This algorithm uses the standard deviation.

        Parameters
        ----------
        target_risk_annual
            the target risk per year

        Returns
        -------
        np.ndarray
            a 2D array of the recommened leverage for trading period
        """
        annualized_volatility = self.market_data.mds.vola_anno

        # Apply smoothing to volatility
        if self.smoothing > 1:
            annualized_volatility = bn.move_mean(annualized_volatility, self.smoothing)

        return target_risk_annual / annualized_volatility

    def _aggressive_sizing(self, risk_limit_per_trade: float) -> np.ndarray:
        """Calculates the maximum leverage based on 'close' prices.

        This is my own algorithm that is very aggressive and
        therefore also more risky. It uses the ATR for estimating
        short-term volatility as basis for the calculation.

        This might be more appropriate for short timeframes, like the
        5m, 15m, or 30m timeframes.

        Parameters:
        ----------
        risk_limit_per_trade: float
            Portion of the available equite to risk per trade.

        Returns
        -------
        np.ndarray
            a 2D array of the recommened leverage for trading period
        """
        return risk_limit_per_trade / (
            self.market_data.mds.atr / self.market_data.close
        )


spec = [
    ("market_data", MarketDataStore.class_type.instance_type),
    ("risk_level", int8),
    ("smoothing", int8),
    ("_valid_risk_levels", int8[:]),
    ("_target_vola", float32[:])
]

@jitclass(spec=spec)
class Leverage:

    def __init__(
            self, 
            market_data: MarketDataStore,
            risk_level: int = 1,
            smoothing: int = 1
    ) -> None:
        self.market_data = market_data
        self.risk_level = risk_level
        self.smoothing = smoothing

        self._valid_risk_levels = np.asarray(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dtype=np.int8
        )
        self._target_vola = np.asarray(
            [0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60],
            dtype=np.float32
        )

    def leverage(self) -> np.ndarray:
        """Calculates the maximum leverage based on 'close' prices.

        This is the method described by Robert Carver in 'Leveraged
        Trading'. The leverage calulation is based on the the ratio
        between the recent annualized volatility and the target 
        volatility (which is determined by the risk level argument
        for the init method of this class).

        These are raw values which might still be too high, depending
        on the market, the leverage allowed by the exchange, and the
        risk appetite.

        If the 'smmothing' parameter was specified as parameter for the
        init method, the annualized volatility will be smoothed 
        accordingly to prevent sudden or too frequent changes in the
        position size(s).

        Returns
        -------
        np.ndarray
            A 2D array of the recommended leverage per asset and  
            trading period.
        """
        
        if self.smoothing > 1:
            annualized_volatility = self.market_data.smooth_it(
                self.market_data.annual_vol,
                self.smoothing
            )
        else:
            annualized_volatility = self.market_data.annual_vol
        
        return np.divide(
            self._target_vola[self.risk_level],
            annualized_volatility, 
        )


if __name__ == "__main__":
    md = MarketData.from_random(1000, 5)
    leverage_calculator = Leverage(market_data=md.mds)