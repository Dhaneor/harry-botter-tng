#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a decorator to convert start and end parameters to timestamps.

By using this decorator, the client can specify dates and relative times
in various formats.

1. Human-readable dates (e.g.: 'January 01, 2019 00:00:00')
2. Relative dates ('now UTC', '1 week ago UTC')
3. Timestamps
4. End as None and start as a negative integer
5. Mixed inputs (e.g.: date string and timestamp)


Created on Sat Dec 16 10:17:23 2024

@author_ dhaneor
"""
import dateparser
import logging
import pytz

from functools import wraps
from typing import Callable, Optional, Dict
from time import time
import datetime

logger = logging.getLogger(f"main.{__name__}")


def timestamp_converter(unit: str = "milliseconds") -> Callable:
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
            interval = kwargs.get('interval')

            _validate_arguments(start, end, interval)

            # Convert start and end to timestamps
            start_ts, end_ts = _convert_to_timestamps(start, end, interval, unit)

            # Replace start and end in args or kwargs
            kwargs['start'] = start_ts
            kwargs['end'] = end_ts

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _validate_arguments(start, end, interval):
    if all(v is None for v in [start, end, interval]):
        raise ValueError(
            "start, end and interval seem to be None. This can happen "
            "if they are not provided as keyword arguments"
        )

    if interval is None and isinstance(start, int) and start < 0:
        raise ValueError(
            "'interval' must be provided when 'start' is a negative integer."
        )

    if start is None:
        raise ValueError("'start' must be provided.")

    if not isinstance(start, (int, str)):
        raise ValueError("'start' must be an integer or a string.")

    if end is not None and not isinstance(end, (int, str)):
        raise ValueError("'end' must be an integer or a string (or None).")


def _convert_to_timestamps(
    start: int | str,
    end: int | str | None,
    interval: str,
    unit: str = "milliseconds",
) -> tuple[int, int]:
    # determine the timestamp for 'now'
    now = int(time() * 1000)

    # find the 'end' timestamp
    match end:
        case str():
            end_ts = int(date_to_milliseconds(end))
        case int():
            end_ts = end
        case None:
            end_ts = now
        case _:
            return 0, 0

    if end_ts > now + 1000:
        logger.warning(
            "The 'end' timestamp '%s' is in the future. "
            "Adjusting to the current time: %s." % (end_ts, now)
        )
        end_ts = now

    # find the 'start' timestamp
    if isinstance(start, str):
        start_ts = int(date_to_milliseconds(start))
    elif isinstance(start, int):
        if start < 0:
            start_ts = int(end_ts - start * interval_to_milliseconds(interval) * -1)
        else:
            start_ts = start
    elif start is None:
        start_ts = int(end_ts - 1000 * (interval_to_milliseconds(interval)))
    else:
        return 0, 0

    if start_ts > end_ts:
        raise ValueError('The start timestamp is after the end timestamp.')

    if end_ts - start_ts < 59_000:
        raise ValueError("The time range is too short (< 1 minute).")

    # convert to milliseconds if parameter 'unit' is milliseconds
    if unit == "seconds":
        logger.debug("Converting start and end timestamps to seconds")
        start_ts = int(str(start_ts)[:10])
        end_ts = int(str(end_ts)[:10])

    return int(start_ts), int(end_ts)


def interval_to_milliseconds(interval: str) -> Optional[int]:
    """Convert a Binance interval string to milliseconds

    Parameters
    ----------
    interval: str
        interval string, e.g.: 1m .. 4h .. 3d .. 1w

    Returns
    -------
    int
        value of interval in milliseconds
    None if interval prefix is not a decimal integer
    None if interval suffix is not one of m, h, d, w

    Raises
    ------
    ValueError
        if the interval prefix is not a decimal integer
    KeyError
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
        logger.error(f"Invalid interval: {interval}")
        return None


def date_to_milliseconds(date_str: str) -> int:
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC",
    "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    Arguments:
    ----------
    date_str: str
        date in readable format, i.e. "January 01, 2018",
        "11 hours ago UTC", "now UTC"
    """
    # get epoch value in UTC
    epoch = datetime.datetime.fromtimestamp(0, tz=datetime.UTC)

    # parse our date string
    d = dateparser.parse(date_str, settings={"TIMEZONE": "UTC"})

    # if the date is not timezone aware apply UTC timezone

    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)
