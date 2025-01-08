#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import numpy as np
import bottleneck as bn
import logging
from sys import getsizeof

from analysis.models.market_data import MarketData
from analysis.statistics.correlation import Correlation
from analysis.util import proj_types as tp  # noqa: F401, E402

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


# ======================================================================================
class DiversificationMultiplier:
    """
    Class to calculate the diversification multiplier.

    'The only free lunch in investing/trading is diversification.' ;)

    The idea is, that with more assets you can take on more risk with
    your single positions because of the diversified risk. The less
    correlated the assets are, the higher the diversification multiplier.

    The multiplier is meant to be applied to:

    1) The leverage/position size that was determined by the position
    sizing algorithm of a strategy which operates on multiple assets.
    The input array should contain the 'close' prices for each asset.

    2) The leverage/position size of multiple strategies for the same
    asset. The input array should contain the the equity/portfolio value
    for the different strategies in this case.

    The matrix in the class definition belwo is used to determine the
    'diversification multiplier'. The key represent the (closest value
    to) the mean correlation of the assets for the lookback period
    (defined by the argument for the init method, default 14). The
    sub-keys stand for the number of assets/strategies. The values are
    the multiplier to apply to the leverage/position size.

    This is taken from Robert Carver´s book: Systematic Trading (p.131)
    """

    DM_MATRIX = {
        0: {2: 1.41, 3: 1.73, 4: 2.0, 5: 2.2, 10: 3.2, 15: 3.9, 20: 4.5, 50: 7.1},
        0.25: {
            2: 1.27,
            3: 1.41,
            4: 1.51,
            5: 1.58,
            10: 1.75,
            15: 1.83,
            20: 1.86,
            50: 1.94,
        },  # noqa: E501
        0.5: {
            2: 1.15,
            3: 1.22,
            4: 1.27,
            5: 1.29,
            10: 1.35,
            15: 1.37,
            20: 1.38,
            50: 1.40,
        },  # noqa: E501
        0.75: {
            2: 1.10,
            3: 1.12,
            4: 1.13,
            5: 1.15,
            10: 1.17,
            15: 1.17,
            20: 1.18,
            50: 1.19,
        },  # noqa: E501
        1.0: {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0, 50: 1.0},
    }

    def __init__(self, data: np.ndarray, period: int = 14):
        self._data = data
        self.period = period
        self.correlation_analyzer = Correlation()
        self._multiplier = None

    def _calculate_rolling_mean_correlation(self):
        return self.correlation_analyzer.rolling(self._data, self.period)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray):
        if not np.array_equal(self._data, new_data):
            self._data = new_data
            self._rolling_mean_correlation = self._calculate_rolling_mean_correlation()
            self._multiplier = self._calculate_multiplier()

    @property
    def multiplier(self) -> np.ndarray:
        if self._multiplier is None:
            self._multiplier = self._calculate_multiplier()
        return self._multiplier

    def _calculate_multiplier(self) -> np.ndarray:
        rolling_mean_correlation = self._calculate_rolling_mean_correlation()

        if self._data.shape[1] < 2:
            return np.ones(self._data.shape[0], dtype=np.float16)

        choices_for_correlations = np.array(list(self.DM_MATRIX.keys()))
        choices_for_no_of_assets = np.array(list(self.DM_MATRIX[0].keys()))

        no_of_assets = min(self._data.shape[1], 50)
        closest_no_of_assets = choices_for_no_of_assets[
            np.argmin(np.abs(choices_for_no_of_assets - no_of_assets))
        ]

        closest_corr_indices = np.searchsorted(
            choices_for_correlations, rolling_mean_correlation, side="left"
        )
        closest_corr_indices = np.clip(
            closest_corr_indices, 0, len(choices_for_correlations) - 1
        )
        closest_corrs = choices_for_correlations[closest_corr_indices]

        result = np.array(
            [self.DM_MATRIX[corr][closest_no_of_assets] for corr in closest_corrs],
            dtype=np.float16,
        )
        return result


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

        self.dmc = DiversificationMultiplier(market_data)

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
                    self.market_dataclose, self.max_leverage, dtype=np.float16
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

            # necessary??? np.nan_to_num(leverage)

            # apply the diversification multiplier if there are multiple assets
            if self.market_data.close.shape[1] > 1:
                lv = np.multiply(lv, self.dmc.multiplier)

            # Apply the maximum allowed leverage
            lv = np.minimum(lv, self.max_leverage).astype(np.float16)

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
        annualized_volatility = self.market_data.mds.annual_vol

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

    def _yield_single(self):  #  -> Generator[np.ndarray, ...]:
        for idx in range(len(self.market_data.open.shape[1])):
            yield (
                self.market_data.open[:, idx],
                self.market_data.high[:, idx],
                self.market_data.low[:, idx],
                self.market_data.close[:, idx],
                self.market_data.volume[:, idx],
            )
