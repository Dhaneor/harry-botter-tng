#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides functions to calculate statistics for backtest results.

Created on Mon Oct 28 21:58:50 2024

@author: dhaneor
"""
import logging
import numpy as np


logger = logging.getLogger("main.statistics")
logger.setLevel(logging.DEBUG)


def calculate_profit(portfolio_values: np.ndarray) -> float:
    """
    Calculate the total profit of the strategy.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.

    Returns:
    -------
    float
        Total profit as a percentage.
    """
    return (portfolio_values[-1] / portfolio_values[0] - 1) * 100


def calculate_annualized_volatility(
    portfolio_values: np.ndarray,
    periods_per_year: int = 365,
    use_log_returns: bool = True
) -> float:
    """
    Calculate the annualized volatility of the portfolio.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.
    periods_per_year : int, optional
        Number of periods in a year (default is 365 for daily data).
    use_log_returns : bool, optional
        Whether to use log returns or simple returns (default is True).

    Returns:
    -------
    float
        Annualized volatility as a percentage.
    """
    if use_log_returns:
        returns = np.log(portfolio_values[1:] / portfolio_values[:-1])
    else:
        returns = portfolio_values[1:] / portfolio_values[:-1] - 1

    return np.std(returns) * np.sqrt(periods_per_year) * 100


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of the strategy.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.

    Returns:
    -------
    float
        Maximum drawdown as a percentage.
    """
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown) * 100


def calculate_sharpe_ratio(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Sharpe ratio of the strategy.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.
    risk_free_rate : float, optional
        The annualized risk-free rate of return (default is 0.0).
    periods_per_year : int, optional
        Number of periods in a year (default is 365 for daily data).

    Returns:
    -------
    float
        Sharpe ratio.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Adjust risk-free rate to the correct time period
    period_risk_free_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - period_risk_free_rate

    # Annualize the mean and standard deviation of excess returns
    annualized_excess_return = np.mean(excess_returns) * periods_per_year
    annualized_volatility = np.std(excess_returns) * np.sqrt(periods_per_year)

    return annualized_excess_return / annualized_volatility \
        if annualized_volatility != 0 else 0


def calculate_sortino_ratio(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Sortino ratio of the strategy.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.
    risk_free_rate : float, optional
        The annualized risk-free rate of return (default is 0.0).
    periods_per_year : int, optional
        Number of periods in a year (default is 365 for daily data).

    Returns:
    -------
    float
        Sortino ratio.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Adjust risk-free rate to the correct time period
    period_risk_free_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - period_risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    # Annualize the mean excess return
    expected_return = np.mean(excess_returns) * periods_per_year

    # Annualize the downside deviation
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) \
        * np.sqrt(periods_per_year)

    return expected_return / downside_deviation if downside_deviation != 0 else np.inf


def calculate_kalmar_ratio(
    portfolio_values: np.ndarray,
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Kalmar ratio of the strategy.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.
    periods_per_year : int, optional
        Number of periods in a year (default is 365 for daily data).

    Returns:
    -------
    float
        Kalmar ratio.
    """
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = \
        (1 + total_return) ** (periods_per_year / len(portfolio_values)) - 1

    # Convert percentage to decimal
    max_drawdown = calculate_max_drawdown(portfolio_values) / 100
    return annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf


def calculate_statistics_min(portfolio_values: np.ndarray) -> dict:
    """
    Calculate all statistics for a given backtest result.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.

    Returns:
    -------
    dict
        Dictionary containing all calculated statistics.
    """
    return {
        'profit': calculate_profit(portfolio_values),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
    }


def calculate_statistics(
    portfolio_values: np.ndarray, risk_free_rate: float = 0, periods_per_year: int = 365
) -> dict:
    """
    Calculate all statistics for a given backtest result.

    Parameters:
    ----------
    portfolio_values : np.ndarray
        1D array of portfolio values over time.

    Returns:
    -------
    dict
        Dictionary containing all calculated statistics.
    """
    return {
        'profit': calculate_profit(portfolio_values),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'sharpe_ratio': calculate_sharpe_ratio(
            portfolio_values, risk_free_rate, periods_per_year
            ),
        'sortino_ratio': calculate_sortino_ratio(
            portfolio_values, risk_free_rate, periods_per_year
            ),
        'kalmar_ratio': calculate_kalmar_ratio(portfolio_values, periods_per_year),
        'annualized_volatility': calculate_annualized_volatility(
            portfolio_values, periods_per_year, use_log_returns=False
            )
    }


# ---------------------------------------- Tests -------------------------------------
def test_calculate_profit(series: np.ndarray) -> None:
    profit = calculate_profit(series)
    expected = 1
    assert np.isclose(profit, expected, atol=1e-6), \
        f"[PR] Expected {expected} but got {np.round(profit, 2)}"


def test_calculate_annualized_volatility(series: np.ndarray) -> None:
    av = calculate_annualized_volatility(series, use_log_returns=False)
    expected = 19  # Approximate value based on the given series length and values.
    assert np.isclose(av, expected, atol=0.1), \
        f"[AV] Expected {expected} but got {np.round(av, 1)}"


def test_calculate_max_drawdown(series: np.ndarray) -> None:
    md = calculate_max_drawdown(series)
    expected = -3
    assert np.isclose(md, expected, atol=0.01), \
        f"[MD] Expected {expected} but got {np.round(md, 2)}"


def test_calculate_sharpe_ratio(series: np.ndarray) -> None:
    sr = calculate_sharpe_ratio(
        portfolio_values=series, risk_free_rate=0, periods_per_year=365
        )
    expected = 1.8297
    assert np.isclose(sr, expected, atol=0.01), \
        f"[SR] Expected {expected} but got {np.round(sr, 4)}"


def test_calculate_sortino_ratio(series: np.ndarray) -> None:
    sr = calculate_sortino_ratio(series, 0, 365)
    expected = 1.83
    assert np.isclose(sr, expected, atol=0.01), \
        f"[SR] Expected {expected} but got {np.round(sr, 3)}"


def test_calculate_kalmar_ratio(series: np.ndarray) -> None:
    kr = calculate_kalmar_ratio(series)
    expected = 11.78
    assert np.isclose(kr, expected, atol=0.01), \
        f"Expected {expected} but got {np.round(kr, 3)}"


def test_calculate_statistics():
    portfolio_values = np.array([100, 99, 98, 97, 98, 99, 100, 101, 102, 103, 102, 101])
    print(calculate_statistics(portfolio_values, 0, 365))

    test_calculate_profit(portfolio_values)
    test_calculate_annualized_volatility(portfolio_values)
    test_calculate_sharpe_ratio(portfolio_values)
    test_calculate_sortino_ratio(portfolio_values)
    test_calculate_kalmar_ratio(portfolio_values)

    print("All tests passed successfully!")


# ============================================ Run Tests =============================
if __name__ == "__main__":
    test_calculate_statistics()
