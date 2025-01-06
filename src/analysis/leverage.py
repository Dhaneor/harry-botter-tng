#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import numpy as np
import bottleneck as bn
import logging

from numba import njit  # noqa: F401, E402
from functools import partial
from sys import getsizeof
from typing import Generator

from analysis.models.market_data import MarketData
from .indicators.indicators_fast_nb import atr
from .util import proj_types as tp  # noqa: F401, E402

logger = logging.getLogger("main.position_size_manager")
logger.setLevel(logging.DEBUG)

"""
The following matrix is used to determine the 'diversification
multiplier'. The key represent the (closest value to) the mean
correlation of the assets. The sub-keys stand for the number
of assets. The values are the multiplier to apply to the position
size.

The idea is, that with more assets you can take on more risk with
your single positions because of the diversified risk. The less
correlated the assets are, the higher the diversification multiplier.

This is taken from Robert Carver´s book: Systematic Trading (p.131)
"""
DM_MATRIX = {
    0: {2: 1.41, 3: 1.73, 4: 2.0, 5: 2.2, 10: 3.2, 15: 3.9, 20: 4.5, 50: 7.1},
    0.25: {2: 1.27, 3: 1.41, 4: 1.51, 5: 1.58, 10: 1.75, 15: 1.83, 20: 1.86, 50: 1.94},   # noqa: E501
    0.5: {2: 1.15, 3: 1.22, 4: 1.27, 5: 1.29, 10: 1.35, 15: 1.37, 20: 1.38, 50: 1.40},   # noqa: E501
    0.75: {2: 1.10, 3: 1.12, 4: 1.13, 5: 1.15, 10: 1.17, 15: 1.17, 20: 1.18, 50: 1.19},   # noqa: E501
    1.0: {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0, 50: 1.0},
}

TRADING_DAYS_PER_YEAR = 365
MILLISECONDS_PER_DAY = 1000 * 60 * 60 * 24
MILLISECONDS_PER_YEAR = TRADING_DAYS_PER_YEAR * MILLISECONDS_PER_DAY

risk_levels = {
    1: 0.12,
}

INTERVAL_IN_MS = {
    '1m': 60 * 1000,
    '3m': 180 * 1000,
    '5m': 300 * 1000,
    '15m': 900 * 1000,
    '30m': 1800 * 1000,
    '1h': 3600 * 1000,
    '2h': 7200 * 1000,
    '4h': 14400 * 1000,
    '6h': 21600 * 1000,
    '12h': 43200 * 1000,
    '1d': 86400 * 1000,
    '3d': 259200 * 1000,
    '1w': 604800 * 1000,
    '1M': MILLISECONDS_PER_YEAR,
}


# --------------------------------------------------------------------------------------
def vol_anno(
    close: np.ndarray,
    interval_in_ms: int,
    lookback: int = 14,
    use_log_returns: bool = True,
) -> np.ndarray:
    """Calculates the annualized volatility based on 'close' prices.

    NOTE: There are a couple schools of thought on which method to
    calculate the returns in this context is more appropriate:

    Log returns help normalize the distribution of returns, since raw
    returns are often right skewed. This may produce more stable
    volatility estimates.
    However, normal returns represent the true percentage change, so
    directly convey volatility. Log returns warp the magnitude of
    changes. In practice, both methods are used:

    Log returns are more common in academic literature as they aid
    statistical analysis. But normal returns are more intuitive and
    align better with how traders think in percentages. So there isn't
    a single standard approach. Using log returns has some statistical
    advantages, but normal returns may better match the reality of
    percentage changes.

    In summary:

    Log returns help normalize, but warp magnitudes. Normal returns
    directly represent percentage changes.
    Since you're building a trading tool, normal returns are likely
    the better choice for more intuitive and representative volatility
    figures. But log returns could be less noisy. It depends on your
    specific requirements!

    Parameters
    ----------
    close
        the close prices
    interval_in_ms
        the trading interval in milliseconds
    lookback
        the lookback period, default is 14
    use_log_returns
        whether to use log returns or not, default is False

    Returns
    -------
    np.ndarray
        the annualized volatility
    """

    periods_per_year = MILLISECONDS_PER_YEAR / interval_in_ms
    out = np.empty_like(close)

    if use_log_returns:
        out[1:] = np.log(close[1:] / close[:-1])
    else:
        out[1:] = close[1:] / close[:-1] - 1

    out[0] = np.nan

    return bn.move_std(out, window=lookback) * np.sqrt(periods_per_year)


@njit(cache=True, fastmath=True, nogil=True)
def vol_anno_nb(close, interval_in_ms, lookback):
    """Numba version of the vol_anno function.

    Note: It's not faster :(
    """
    out = np.empty_like(close)
    factor = np.sqrt(MILLISECONDS_PER_YEAR / interval_in_ms)

    for i in range(lookback, len(close)):
        out[i] = np.std(close[i - lookback: i]) * factor

    return out


def _interval_in_ms(data):
    x = data["open time"]

    return int(np.min(np.diff(x[~np.isnan(x)])))


def _aggressive_sizing(
    data: dict,
    risk_limit_per_trade: float,
    max_leverage: float
) -> np.ndarray:
    """Calculates the maximum leverage based on 'close' prices.

    This is my own algorithm that is very aggressive and
    therefore also more risky. It uses the ATR for estimating
    short-term volatility as basis for the calculation.

    :param df: a DataFrame with at least a 'close' colmun
    :type df: pd.DataFrame
    :return: the DataFrame with added column 'leverage'
    :rtype: np.ndarray
    """
    atr_ = atr(data["open"], data["high"], data["close"])

    leverage = risk_limit_per_trade / (atr_ / data["close"])

    # Apply maximum leverage limit
    return np.minimum(leverage, max_leverage)


def _conservative_sizing(
    data: dict,
    interval_in_ms: int,
    max_leverage: float,
    target_risk_annual: float,
    smoothing: int = 1
) -> np.ndarray:
    """Calculates the maximum leverage based on 'close' prices.

    This is the method described by Robert Carver in 'Leveraged
    Trading'. It is very conservative, at least when the account
    level risk target is also calculated by the method(s) described
    in the book (he proposes to target 12% annualized volatility
    for the portfolio).

    This algorithm uses the standard deviation.

    Parameters
    ----------
    close
        the close prices
    target_risk_annual
        the target risk per year
    smoothing
        the smoothing period (less erratic results)
    interval
        the trading interval in milliseconds

    Returns
    -------
    np.ndarray
        the maximum leverage
    """
    annualized_volatility = vol_anno_nb(
        close=data,
        interval_in_ms=interval_in_ms,
        lookback=21,
        # use_log_returns=False
    )

    # Apply smoothing to volatility
    if smoothing > 1:
        annualized_volatility = bn.move_mean(annualized_volatility, smoothing)

    # Calculate leverage
    leverage = target_risk_annual / annualized_volatility



    return np.nan_to_num(leverage)


# ======================================================================================
run_funcs = {
    1: partial(_conservative_sizing, target_risk_annual=0.12),
    2: partial(_conservative_sizing, target_risk_annual=0.18),
    3: partial(_conservative_sizing, target_risk_annual=0.24),
    4: partial(_conservative_sizing, target_risk_annual=0.30),
    5: partial(_conservative_sizing, target_risk_annual=0.36),
    6: partial(_conservative_sizing, target_risk_annual=0.42),
    7: partial(_conservative_sizing, target_risk_annual=0.48),
    8: partial(_conservative_sizing, target_risk_annual=0.54),
    9: partial(_conservative_sizing, target_risk_annual=0.60),
    10: partial(_aggressive_sizing, risk_limit_per_trade=0.01),
    11: partial(_aggressive_sizing, risk_limit_per_trade=0.03),
    12: partial(_aggressive_sizing, risk_limit_per_trade=0.05),
}


def calculate_leverage(
    data: np.ndarray,
    interval_in_ms: int,
    max_leverage: float,
    risk_level: int = 1
) -> np.ndarray:
    """Calculates the maximum leverage based on 'close' prices.

    Parameters
    ----------
    data
        OHLCV dictionary

    risk_level
        the risk level, default is 1

    Returns
    -------
    np.ndarray
        the maximum/recommended leverage

    Raises
    ------
    KeyError
        if the risk level is not valid
    """
    return run_funcs[risk_level](
        data=data,
        interval_in_ms=interval_in_ms,
        max_leverage=max_leverage
        )


def valid_risk_levels() -> tuple[int]:
    return tuple(run_funcs.keys())


# ======================================================================================
def mean_correlation(close_prices: np.ndarray, period: int) -> float:
    """Returns the mean correlation between the close prices.

    Parameters
    ----------
    close_prices
        the close prices, one column for each asset
    period
        the lookback period

    Returns
    -------
    float
        the mean correlation
    """

    # Calculate correlation matrix
    correlations = np.corrcoef(close_prices[-period:], rowvar=False)

    # Set diagonal to NaN
    np.fill_diagonal(correlations, np.nan)

    # Take mean of upper triangle
    return np.nanmean(correlations[np.triu_indices(len(correlations), k=1)])


def diversification_multiplier(
    close_prices: tp.Array_2d, period: int = 14
) -> float:
    """Calculates the diversification multiplier based on 'close' prices.

    This is the method described by Robert Carver in 'Leveraged
    Trading'. It is very conservative, at least when the account
    level risk target is also calculated by the method(s) described
    in the book (he proposes 12%).
    This algorithm uses the standard deviation.

    Parameters
    ----------
    close_prices
        the close prices, one column for each asset
    period
        the lookback period

    Returns
    -------
    float
        the diversification multiplier
    """

    # multiplier is always '1' for one asset
    if close_prices.shape[1] < 2:
        return 1

    # find the closest correlation value in our matrix (see at top of
    # file) to the actual mean correlation between the assets in the
    # lookback period
    choices_for_correlations = DM_MATRIX.keys()

    closest_corr = min(
        choices_for_correlations,
        key=lambda x: abs(x - mean_correlation(close_prices, period)),
    )

    # find the closest number of assets in our matrix to the actual
    # number of assets
    choices_for_no_of_assets = DM_MATRIX[0].keys()
    no_of_assets = 50 if close_prices.shape[1] > 50 else close_prices.shape[1]

    closest_no_of_assets = min(
        choices_for_no_of_assets, key=lambda x: abs(x - no_of_assets)
    )

    return DM_MATRIX[closest_corr][closest_no_of_assets]


class LeverageCalculator:

    def __init__(
        self,
        market_data: MarketData,
        risk_level: int = 1,
        max_leverage: float = 1.0,
        atr_window: int = 21):
        self.market_data = market_data

        self._validate_risk_level(risk_level)
        self.risk_level = risk_level
        self.max_leverage = max_leverage

        self.interval = market_data.interval
        self.interval_in_ms = market_data.interval_in_ms

        self.atr_window = atr_window

        self._cache = {}

        # pre-populate the cache ...
        if len(market_data) < 5_000:
            # ...for all risk levels for smaller datasets
            for risk_level in valid_risk_levels():
                self.leverage(risk_level)
        else:
            #...for the current risk level for larger datasets
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
        self._validate_risk_level(risk_level)

        if self._cache.get(risk_level, None) is None:
            lv = run_funcs[risk_level](
                data=self.market_data,
                interval_in_ms=self.interval_in_ms,
                max_leverage=self.max_leverage
            )

            if getsizeof(self.cache) > 100_000_000:
                self._cache.clear()
                logger.info("Cache cleared due to high memory usage.")

            self._cache[risk_level] = np.minimum(lv, self.max_leverage)

        return self._cache[risk_level]

    def _validate_risk_level(self, risk_level: int) -> None:
        if risk_level not in valid_risk_levels():
            raise ValueError(
                f"Invalid risk level: {risk_level}. Valid risk levels: "
                f"{valid_risk_levels()}"
                )

    def _yield_single(self) -> Generator[np.ndarray, ...]:
        for idx in range(len(self.market_data.open.shape[1])):
            yield (
                self.market_data.open[:, idx],
                self.market_data.high[:, idx],
                self.market_data.low[:, idx],
                self.market_data.close[:, idx],
                self.market_data.volume[:, idx],
            )
