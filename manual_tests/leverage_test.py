#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import time
import numpy as np

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

from analysis import leverage as lv  # noqa: E402, F401
from analysis.indicators.indicators_fast_nb import atr  # noqa: E402, F401
from util import execution_time  # noqa: E402, F401

length = 1_000

ohlcv = {
    'open': np.random.rand(length),
    'high': np.random.rand(length),
    'low': np.random.rand(length),
    'close': np.random.rand(length),
    'volume': np.random.rand(length),
}

atr = atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], 14)


# ======================================================================================
def change_atr_period(new_atr_period: int):
    raise NotImplementedError("should be able to change the ATR period!")


@execution_time
def get_max_leverage():
    return lv.max_leverage(data=ohlcv, risk_level=3)


@execution_time
def get_diversifcation_multiplier(close_prices: np.ndarray):
    return lv.diversification_multiplier(close_prices=close_prices)


# --------------------------------------------------------------------------------------
@execution_time
def test_vol_anno(close: np.ndarray):
    ohlcv['volatility'] = lv.vol_anno(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
        use_log_returns=False
    )


@execution_time
def test_vol_anno_nb(close: np.ndarray):
    ohlcv['volatility'] = lv.vol_anno_nb(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
    )


@execution_time
def test_conservative_sizing(data: dict):
    lv._conservative_sizing(data=data, target_risk_annual=0.1, smoothing=1)


@execution_time
def test_aggressive_sizing(data: dict):
    lv._aggressive_sizing(data=data, risk_limit_per_trade=0.05)


@execution_time
def test_max_leverage_fast(data: dict):
    ohlcv['leverage'] = lv.max_leverage(data=data, risk_level=1)


def test_get_diversification_multiplier(period: int = 30, number_of_assets: int = 3):
    close_prices = np.random.randint(100, 1000, size=(period, number_of_assets))

    for _ in range(5):
        dm = get_diversifcation_multiplier(close_prices=close_prices)

    print(f"diversification multiplier: {round(dm,2)}")


# -----------------------------------------------------------------------------
#                                   MAIN                                      #
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # for _ in range(5):
    #     test_vol_anno_nb(ohlcv['close'])

    # for _ in range(5):
    #     test_vol_anno(ohlcv['close'])

    # for _ in range(5):
    #     test_conservative_sizing(ohlcv)

    # for _ in range(5):
    #     test_aggressive_sizing(ohlcv)

    # for _ in range(5):
    #     test_max_leverage_fast(ohlcv)

    # df = pd.DataFrame.from_dict(ohlcv)
    # print(df.tail(10))

    # test_get_diversification_multiplier(60, 5)

    # =================================================================================
    runs = 10_000
    length = 1000
    data = np.random.rand(length)
    func = lv.calculate_leverage

    func(data, 86400, 3)

    start = time.perf_counter()
    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            func(data, 86400, 3)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    # for _ in range(runs):
    #     test_strategy_run(s, False)

    et = time.perf_counter()
    print(f"length data: {length}")
    print(f"average execution time: {((et - start)*1_000_000/runs):.2f} microseconds")
    print(f"iterations per second: {runs / ((et - start)):.2f} iterations/second")
