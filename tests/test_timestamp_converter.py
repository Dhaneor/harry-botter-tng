#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:17:23 2024

@author_ dhaneor
"""
import os
import pytest
import pytz
import sys
from datetime import datetime, timedelta
from time import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data.rawi.util.timestamp_converter import timestamp_converter  # noqa: E402


# Helper function to get current timestamp in milliseconds
def now_ms():
    return int(time() * 1000)


# Test function to be decorated
@timestamp_converter(unit='milliseconds')
def example_function(start, end, interval=None):
    return start, end


def test_human_readable_dates():
    start, end = example_function(
        start="January 01, 2019 00:00:00", end="January 02, 2019 00:00:00"
    )
    assert start == 1546300800000
    assert end == 1546387200000


def test_relative_dates():
    now = now_ms()
    start, end = example_function(start="1 week ago UTC", end="now UTC")
    assert (
        now - 7 * 24 * 60 * 60 * 1000 - 1000
        <= start
        <= now - 7 * 24 * 60 * 60 * 1000 + 1000
    )
    assert now - 1000 <= end <= now + 1000


def test_timestamps():
    start, end = example_function(start=1546300800000, end=1546387200000)
    assert start == 1546300800000
    assert end == 1546387200000


def test_end_none_start_negative():
    now = now_ms()
    start, end = example_function(start=-24, end=None, interval='1h')
    assert now - 24 * 60 * 60 * 1000 - 1000 <= start <= now - 24 * 60 * 60 * 1000 + 1000
    assert now - 1000 <= end <= now + 1000


def test_mixed_inputs():
    now = now_ms()
    start, end = example_function(start="January 01, 2019 00:00:00", end=now)
    assert start == 1546300800000
    assert end == now


def test_invalid_date_string():
    with pytest.raises(AttributeError):
        example_function(start="Invalid Date", end="now UTC")


def test_future_end_date():
    now = now_ms()
    future = datetime.now() + timedelta(days=30)
    future_str = future.strftime("%B %d, %Y %H:%M:%S")
    _, end = example_function(start=now - 120_000, end=future_str)
    assert now_ms() - 1000 <= end <= now_ms() + 1000


def test_different_intervals():
    @timestamp_converter()
    def daily_function(start, end, interval):
        return start, end

    start, end = daily_function(start=-7, end=None, interval='1d')
    now = now_ms()
    assert (
        now - 7 * 24 * 60 * 60 * 1000 - 1000
        <= start
        <= now - 7 * 24 * 60 * 60 * 1000 + 1000
    )
    assert now - 1000 <= end <= now + 1000


def test_milliseconds_vs_seconds():
    @timestamp_converter(unit="seconds")
    def seconds_function(start, end, interval="1h"):
        return start, end

    start_ms, end_ms = example_function(
        start="January 01, 2019 00:00:00", end="January 02, 2019 00:00:00"
    )
    start_s, end_s = seconds_function(
        start="January 01, 2019 00:00:00", end="January 02, 2019 00:00:00"
    )

    assert start_ms == start_s * 1000
    assert end_ms == end_s * 1000


def test_relative_time_expressions():
    for unit in ["minute", "hour", "day", "week", "month", "year"]:
        now = datetime.now(pytz.utc)
        now_ms = int(now.timestamp() * 1000)

        start, end = example_function(start=f"1 {unit} ago UTC", end="now UTC")

        if unit == "minute":
            expected_start = now - timedelta(minutes=1)
        if unit == "hour":
            expected_start = now - timedelta(hours=1)
        elif unit == "day":
            expected_start = now - timedelta(days=1)
        elif unit == "week":
            expected_start = now - timedelta(weeks=1)
        elif unit == "month":
            expected_start = now - timedelta(days=30)  # Approximate
        elif unit == "year":
            expected_start = now.replace(year=now.year - 1)

        expected_start_ms = int(expected_start.timestamp() * 1000)

        assert expected_start_ms - 1000 <= start <= expected_start_ms + 1000, \
            (
                f"Start time for '1 {unit} ago UTC' is incorrect. "
                f"Expected {expected_start_ms} "
                f"but got {start} (difference: {expected_start_ms - start})"
            )
        assert now_ms - 2000 <= end <= now_ms + 2000


if __name__ == "__main__":
    pytest.main([__file__])
