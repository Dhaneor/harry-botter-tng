#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 04 15:17:53 2022

@author: dhaneor
"""
import os
import sys
import time
from pprint import pprint
from cProfile import Profile
from pstats import SortKey, Stats

# ------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from src.exchange.util import ohlcv_download_prepper as odp  # noqa: E402
from util.timeops import execution_time  # noqa: E402

dp = odp.OhlcvDownloadPrepper()


# ==============================================================================
def test_prepare_single():
    symbol = "XRP-USDT"
    intervals = ["1h", "6h", "12h", "1d"]
    start = -1000  # 2021-01-01 00:00:00"
    end = None  # "2023-12-31 00:00:00"

    @execution_time
    def execute(symbol, interval, start, end):
        return dp._prepare_request(symbol, interval, start, end)

    for interval in intervals:
        try:
            pprint(execute(symbol, interval, start, end))
        except Exception as e:
            print(e)


# ==============================================================================
if __name__ == "__main__":
    test_prepare_single()

    sys.exit()

    runs = 1_000
    st = time.time()

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            dp._prepare_request(
                "XRP-USDT", "1d", -1000, None
                )

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(40)

    )

    print(
        f"execution time: {((time.time() - st) * 1_000_000 / runs):.2f} microseconds"
        )
