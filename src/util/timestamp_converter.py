#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a decorator to convert start and end parameters to timestamps.

Created on Sat Dec 16 10:17:23 2024

@author_ dhaneor
"""
import re

from functools import wraps
from typing import Union, Callable, Optional, Dict
from time import time
from datetime import datetime, timezone, timedelta


def timestamp_converter(
    interval: str,
    unit: str = "milliseconds",
) -> Callable:
    """
    A decorator to convert start and end parameters to timestamps.

    Parameters
    ----------
    interval : str
        The time interval (e.g., '1h', '1d').
    unit : str, optional
        Return as 'milliseconds' or 'seconds'. Default is "milliseconds".
    verbose : bool, optional
        Print verbose output or not. Default is False.

    Returns
    -------
    Callable
        A decorator function.

    Notes
    -----
    This decorator converts 'start' and 'end' parameters of the decorated function
    to timestamps, making it versatile for different use cases.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract start and end from args or kwargs
            start = kwargs.get('start')
            end = kwargs.get('end')

            # Convert start and end to timestamps
            start_ts, end_ts = _convert_to_timestamps(start, end, interval, unit)

            # Replace start and end in args or kwargs
            kwargs['start'] = start_ts
            kwargs['end'] = end_ts

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _convert_to_timestamps(
    start: Union[int, str],
    end: Union[int, str],
    interval: str,
    unit: str = "milliseconds",
) -> tuple[int, int]:
    # determine the timestamp for 'now'
    now = int(time() * 1000)

    # find the 'end' timestamp
    if isinstance(end, str):
        end_ts = int(date_to_milliseconds(end) / 1000)
    elif isinstance(end, int):
        end_ts = end
    elif end is None:
        end_ts = now
    else:
        return 0, 0

    end_ts = min(end_ts, now)

    # find the 'start' timestamp
    if isinstance(start, str):
        start_ts = int(date_to_milliseconds(start) / 1000)
    elif isinstance(start, int):
        if start < 0:
            interval_in_sec = interval_to_milliseconds(interval) / 1000
            start_ts = int(end_ts - start * interval_in_sec * -1)
        else:
            start_ts = start
    elif start is None:
        interval_in_sec = interval_to_milliseconds(interval) / 1000
        start_ts = int(end_ts - 1000 * (interval_in_sec))
    else:
        return 0, 0

    # convert to milliseconds if parameter 'unit' is milliseconds
    if unit == "milliseconds":
        start_ts *= 1000
        end_ts *= 1000

    return int(start_ts), int(end_ts)


def interval_to_milliseconds(interval: str) -> Optional[int]:
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string, e.g.: 1m .. 4h .. 3d .. 1w

    :return:
         int value of interval in milliseconds
         None if interval prefix is not a decimal integer
         None if interval suffix is not one of m, h, d, w

    """
    seconds_per_unit: Dict[str, int] = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }

    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None


def date_to_milliseconds(date_str: str) -> int:
    """
    Parse a date string to milliseconds since epoch.

    Supports various date formats including:
    - ISO 8601
    - Relative time expressions (e.g., '1 day ago', '2 hours ago')
    - 'now UTC'

    Parameters
    ----------
    date_str : str
        The date string to parse.

    Returns
    -------
    int
        Milliseconds since epoch.
    """
    if date_str.lower() == 'now utc':
        return int(time() * 1000)

    # Check for relative time expressions
    relative_match = re.match(
        r'(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago',
        date_str,
        re.IGNORECASE
        )
    if relative_match:
        num, unit = relative_match.groups()
        num = int(num)
        now = datetime.now(timezone.utc)
        if unit.lower() == 'minute':
            delta = now - timedelta(minutes=num)
        elif unit.lower() == 'hour':
            delta = now - timedelta(hours=num)
        elif unit.lower() == 'day':
            delta = now - timedelta(days=num)
        elif unit.lower() == 'week':
            delta = now - timedelta(weeks=num)
        elif unit.lower() == 'month':
            delta = now - timedelta(days=num*30)  # Approximation
        elif unit.lower() == 'year':
            delta = now - timedelta(days=num*365)  # Approximation
        return int(delta.timestamp() * 1000)

    # Try parsing as ISO 8601
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    # If all else fails, raise an error
    raise ValueError(f"Unable to parse date string: {date_str}")
