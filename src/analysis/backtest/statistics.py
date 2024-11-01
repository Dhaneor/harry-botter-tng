#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides functions to calculate statistics for backtest results.

Created on Mon Oct 28 21:58:50 2024

@author: dhaneor
"""
import logging
import numpy as np


logger = logging.getLogger("main.backtest_stats")
logger.setLevel(logging.ERROR)


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
        The risk-free rate of return (default is 0.0).
    periods_per_year : int, optional
        Number of periods in a year (default is 365 for daily data).

    Returns:
    -------
    float
        Sharpe ratio.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


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
        The risk-free rate of return (default is 0.0).
    periods_per_year : int, optional
        Number of periods in a year (default is 252 for daily data).

    Returns:
    -------
    float
        Sortino ratio.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    expected_return = np.mean(excess_returns) * periods_per_year
    downside_deviation = np.sqrt(
        np.mean(downside_returns**2)
        ) * np.sqrt(periods_per_year)
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


def calculate_statistics(portfolio_values: np.ndarray) -> dict:
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
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values),
        'sortino_ratio': calculate_sortino_ratio(portfolio_values),
        'kalmar_ratio': calculate_kalmar_ratio(portfolio_values)
    }
