#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=unused-import
"""
Utility Package for Harry Botter TNG

This package provides various utility functions and classes used throughout
the Harry Botter TNG project. It includes modules for accounting, time operations,
formatting, health checks, and more.

Usage:
    from util import calculate_profit, format_currency
    profit = calculate_profit(100, 120)
    formatted_profit = format_currency(profit)

See individual module documentation for more detailed information on available
functions and classes.

Created on Thu Sep 22 13:00:23 2021

@author_ dhaneor
"""

__version__ = "0.1.0"

from .accounting import Accounting
from .fibonacci import fibonacci, fibonacci_series
from .health import Health
from .log_execution_time import log_execution_time
from .logger_setup import get_logger
from .singleton import SingletonMeta
from .timed_lru_cache import timed_lru_cache
from .timeops import (
    execution_time,
    date_to_milliseconds,
    interval_to_milliseconds,
    unix_to_utc,
    utc_to_unix,
    now_utc_ts,
    get_start_and_end_timestamp,
    seconds_to,
)

__all__ = [
    "Accounting",
    "fibonacci",
    "fibonacci_series",
    "Health",
    "log_execution_time",
    "get_logger",
    "SingletonMeta",
    "timed_lru_cache",
    "execution_time",
    "date_to_milliseconds",
    "interval_to_milliseconds",
    "unix_to_utc",
    "utc_to_unix",
    "now_utc_ts",
    "get_start_and_end_timestamp",
    "seconds_to",
]
