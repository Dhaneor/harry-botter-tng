#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:11:33 2021

@author dhaneor
"""
import datetime
import sys
import os
import pandas as pd
import logging
from pprint import pprint
from random import choice

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger("main")
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)


# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from src.data.ohlcv import OHLCVData, OHLCVTable  # noqa: E402, F401
from src.util.timeops import execution_time  # noqa: E402


@execution_time
def get_test_data(length: int = 1000, interval: str = "15min"):
    """get random OHLCV data for testing purposes."""

    # find the timestamp for the most recent 15 minute data point
    now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = now.timestamp()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Current time: {now_str}")

    most_recent_15m_ts = (now_ts // 900) * 900
    most_recent_15m = (
        datetime
        .datetime.
        fromtimestamp(most_recent_15m_ts, datetime.timezone.utc)
        )
    most_recent_15m_str = most_recent_15m.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Most recent 15 minute data point: {most_recent_15m_str}")

    # create timestamps for <length> data points,
    # ending at current time
    timestamps = pd.date_range(
        start=most_recent_15m - datetime.timedelta(minutes=length * 15),
        periods=length,
        freq="15min",
    ).tolist()

    # create random OHLCV data for each timestamp
    ohlcv_data = [
        {
            "open time": ts,
            "open": choice(range(100, 200)),
            "high": choice(range(100, 200)),
            "low": choice(range(100, 200)),
            "close": choice(range(100, 200)),
            "volume": choice(range(1000, 2000)),
        }
        for ts in timestamps
    ]

    # create DataFrame from OHLCV data and return
    df = pd.DataFrame(ohlcv_data)

    # print the first few rows of the DataFrame
    logger.info("First few rows of OHLCV data:")
    pprint(df.head())

    # print the last few rows of the DataFrame
    logger.info("Last few rows of OHLCV data:")
    pprint(df.tail())

    # print the total number of rows in the DataFrame
    logger.info(f"Total number of rows in OHLCV data: {len(df)}")

    return df


def test_determine_interval():
    # determine and print the interval of the OHLCV data
    interval = OHLCVData("symbol", get_test_data())._determine_interval()
    logger.info(f"Interval of OHLCV data: {interval}")


# ------------------------------------------------------------------------------------
def test_is_up_to_date(symbol_name: str, interval: str):
    table = OHLCVTable(symbol_name, interval)
    logger.info("%s - is_up_to_date: %s", table.name, table.is_up_to_date())


if __name__ == "__main__":
    symbol_name = "BTCUSDT"
    interval = "1h"
    symbols = ["BTCUSDT", "ADAUSDT", "XRPUSDT", "XLMUSDT"]
    intervals = ["5m", "15m", "30m", "1h", "2h", "4h"]

    dates = [
        "1 year ago UTC",
        "now UTC",  # 'October 19, 2023 00:00:00'
    ]
    exchange = "kucoin" if "-" in symbol_name else "binance"

    # test_determine_interval()
    test_is_up_to_date(symbol_name, interval)

    for symbol in symbols:
        for interval in intervals:
            for date in dates:
                table = OHLCVTable(symbol, interval)
                logger.info(
                    "%s - %s - %s - is_up_to_date: %s",
                    table.name,
                    table.symbol,
                    table.interval,
                    table.is_up_to_date(),
                )
