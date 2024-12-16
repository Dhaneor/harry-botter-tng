#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:28:11 2021

@author: dhaneor
"""
from datetime import datetime, timezone
from time import time
from typing import Union, Optional, Dict, Tuple

import dateparser
import pytz
import math


# -----------------------------------------------------------------------------
# decorator function to calculate the execution time for the given function
def execution_time(func):

    def inner(*args, **kwargs):
        _start = time()
        res = func(*args, **kwargs)
        et = seconds_to(time() - _start)
        print(f"<<< execution time for {func.__name__.upper()}:\t{et} >>>")
        return res

    return inner


# -----------------------------------------------------------------------------
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
    epoch: datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d: Optional[datetime] = dateparser.parse(date_str, settings={"TIMEZONE": "UTC"})

    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


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


# -----------------------------------------------------------------------------
# functions to transform (date)time between unix timetamps and human readable
# string format
def unix_to_utc(unix: int) -> str:
    """This function takes a unix timestamp and returns the time in human
    readable form

    :param unix: time as unix timestamp
    :type unix: int

    :returns: converted time as string
    """
    unix = int(unix)
    # check if we got seconds or milliseconds
    if len(str(unix)) == 13:
        unix /= 1000

    try:
        humanDate = datetime.utcfromtimestamp(int(unix)).strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print("Conversion to UTC failed: ", e, " :: ", unix)
        return "1900-01-01 00:00:00"

    return humanDate


def utc_to_unix(date_str):
    """This function takes a a human readable time and returns unix timestamp

    :param date_str: human readable time in the format '%Y-%M-%D %hh:%mm:%ss'
    :type date_str: str

    .. code:: python

        utc_to_unix('2021-12-31 08:00:00')

    :returns: unix timestamp (int)
    """

    error_1 = None

    try:
        date_obj = datetime.strptime(date_str, "%B %d, %Y %H:%M:%S")
        timestamp = date_obj.replace(tzinfo=timezone.utc).timestamp()
    except Exception as e:
        error_1 = f"[timeops] Conversion to timestamp failed: {e} for {date_str}"

    if error_1 is not None:

        try:

            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            timestamp = date_obj.replace(tzinfo=timezone.utc).timestamp()

        except Exception as e:

            error_2 = f"[timeops] Conversion to timestamp failed: {e} for {date_str}"
            print(error_1)
            print(error_2)
            return int(time())

    return int(timestamp * 1000)


def utc_timestamp():
    dt = datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    return utc_time.timestamp()


# -----------------------------------------------------------------------------
# flexible function to convert a start and an end date to timestamps
def get_start_and_end_timestamp(
    start: Union[int, str],
    end: Union[int, str],
    interval: str,
    unit: str = "milliseconds",
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Convert two values (start/end) to seconds or milliseconds.

    Parameters
    ----------
    start : int or str
        The beginning of the time period which can be given as integer or a string.
        There are multiple ways to pass this value:
        - int: If the number is smaller than 10000, it will be interpreted as
                number of intervals counted backwards from the end date.
                Otherwise, it will be treated as a timestamp.
        - str: A string can be in one of many formats that represent a date (and time).
                It is also possible to use expressions like '30 days ago' or 'now UTC'.

    end : int or str
        The end of the time period which can have all the same formats as 'start'
        (except smaller integers for periods).

    interval : str
        The time interval (e.g., '1h', '1d').

    unit : str, optional
        Return as 'milliseconds' or 'seconds'. Default is "milliseconds".

    verbose : bool, optional
        Print verbose output or not. Default is False.

    Returns
    -------
    tuple of int
        A tuple with two timestamps (start_ts, end_ts).
    """

    # determine the timestamp for 'now'
    now = int(time() * 1000)

    # find the 'end' timestamp
    match end:
        case str():
            end_ts = int(date_to_milliseconds(end) / 1000)
        case int():
            end_ts = end
        case None:
            end_ts = now
        case _:
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

    if verbose:
        print(" ")
        print("-" * 80)
        print(f"{start_ts=} ({start}) -> {unix_to_utc(start_ts)}")
        print(f"{end_ts=} ({end}) -> {unix_to_utc(end_ts)}")
        print("-" * 80)
        print(f"interval {interval} = {interval_to_milliseconds(interval)}\n")

    return int(start_ts), int(end_ts)


# -----------------------------------------------------------------------------
# functions to help with the calculation of time differences and their
# representation


def seconds_to(seconds: float) -> str:
    """Convert seconds to a readable string giving the number of days, hours,
    minutes and seconds

    :param seconds: number of seconds to convert
    :type seconds: int or float

    :returns: string

    .. code: python
       '5d 9h 14m 24s'
    """
    res = ""

    # convert seconds to day, hour, minutes and seconds
    days = int(seconds // (24 * 3_600))
    if days:
        res += f"{days}d "

    time_mod = seconds % (24 * 3_600)
    hours = int(time_mod // 3_600)
    if hours:
        res += f"{hours}h "

    time_mod %= 3_600
    minutes = int(time_mod // 60)
    if minutes != 0:
        res += f"{minutes}m "

    time_mod %= 60
    seconds = math.floor(time_mod)
    if seconds != 0:
        res += f"{seconds}s "

    # only show milliseconds for small seconds values
    if not days and not hours and not minutes and seconds < 5:

        time_mod = (time_mod * 1_000) % 1_000
        milliseconds = math.floor(time_mod)

        if milliseconds:
            res += f"{milliseconds}ms "

        # only display microseconds for small millisecond values
        if seconds == 0 and milliseconds < 5:

            time_mod %= 1_000
            microseconds = math.floor(time_mod * 1_000 % 1_000)

            if microseconds:
                res += f"{microseconds}µs "

            # only display nanoseconds for small microsecond values
            if milliseconds == 0 and microseconds < 5:
                res += f"{math.floor(time_mod * 1_000_000 % 1_000)}ns"

    return res


def time_difference_in_ms(
    start: float, end: Optional[float] = None, precision: int = 2
):
    """Calculate the time differende in ms for the interval between start and
    end. If no end is given, the current time will be used

    :param start: the beginning of the interval
    :type start: int

    :param end: the end of the interval
    :type end: int

    :returns: the time difference in ms (int)"""

    if end is None:
        end = time()
    return round((end - start) * 1000, precision)


def seconds_to_next_full(end: str):
    _now = datetime.fromtimestamp(int(time()))

    if end == "minute":
        return 60 - _now.second

    if end == "hour":
        _minutes = 60 - _now.minute
        return _minutes * 60 + 60 - _now.second


# ----------------------------------------------------------------------------
#                               TEST FUNCTIONS                               #
# ----------------------------------------------------------------------------
# test functions
def test_interval_to_milliseconds():
    intervals = ("1m", "15m", "6h", "12h", "1d")
    [print(interval_to_milliseconds(int)) for int in intervals]


def test_seconds_to():
    print("a) longer time difference:")
    # seconds = time() - (time() - (3600*24) * 5.385)
    seconds = 0.5 * 3600 + 23 + 0.5
    print(seconds_to(seconds))

    print("\nb) shorter time difference:")

    seconds = 5.385
    print(seconds_to(seconds))

    print("\nc) very short time difference:")
    seconds = 0.000000385698
    print(seconds_to(seconds))


def test_seconds_to_next_full(end: str):
    seconds = seconds_to_next_full(end)
    print(f"seconds to next full {end}: {seconds}")


def test_decorator():

    _start = time() - 1_000_000
    print(time_difference_in_ms(start=_start))


@execution_time
def test_date_to_milliseconds():

    dates = [
        "February 01, 2022 00:00:00",
        "2022-03-01 00:00:00",
        "UTC now",
        "now UTC",
        "2 weeks ago",
        "30 days ago",
    ]

    for date in dates:
        try:
            unix_epoch = date_to_milliseconds(date)
        except Exception as e:
            print(e)
        reverted = unix_to_utc(unix_epoch)
        print(f"{date} -> {unix_epoch} -> {reverted}")
        print("-" * 80)
    print("\n")


def test_utc_timestamp():
    ts = utc_timestamp()
    print(ts, unix_to_utc(ts))


def test_utc_to_unix(utc_time: str):
    unix = utc_to_unix(utc_time)
    print(f"{utc_time=} :: {unix=}")


def test_start_end_timestamps():
    periods = [
        ("2021-01-01 00:00:00", "2022-01-01 00:00:00"),
        (-1000, "now UTC"),
        (-2000, None),
        (0, "now UTC"),
        ("January 01, 2021 00:00:00", "January 01, 2022 00:00:00"),
        ("January 01, 2021 00:00:00", "now UTC"),
        ("January 01, 2021 00:00:00", int(time())),
        (152997839999, 1530014399999),
    ]

    print("=" * 80)
    [
        get_start_and_end_timestamp(p[0], p[1], interval="1h", verbose=True)
        for p in periods
    ]

    print("\n\n")


def test_start_end_timestamp_for_intervals():
    p = ("2021-01-01 00:00:00", "2021-12-31 00:00:00")
    intervals = ["1m", "15m", "1h", "2h", "6h", "12h", "1d"]

    print("=" * 80)
    [
        get_start_and_end_timestamp(p[0], p[1], interval=int, verbose=True)
        for int in intervals
    ]

    print("\n\n")


# -----------------------------------------------------------------------------
#                                       MAIN                                  #
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # [test_decorator() for _ in range(10)]

    # test_interval_to_milliseconds()

    # test_seconds_to()

    # test_seconds_to_next_full('minute')

    # test_date_to_milliseconds()

    # test_utc_timestamp()

    # test_utc_to_unix('January 01, 2017 00:00:00')

    # print(unix_to_utc(1646514829790))

    test_start_end_timestamps()

    # test_start_end_timestamp_for_intervals()
