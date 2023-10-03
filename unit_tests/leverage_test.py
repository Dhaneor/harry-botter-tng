#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import numpy as np

# ------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from src.analysis import leverage as pm  # noqa: E402, F401
from unit_tests.helpers_ import get_sample_data, get_ohlcv  # noqa: E402, F401
from src.helpers.timeops import execution_time  # noqa: E402, F401


data = get_sample_data(length=1000, interval='15min')

ohlcv = get_ohlcv(symbol='BTCUSDT', interval='1d')

ohlcv_dict = dict(
    interval_in_ms=int(np.min(ohlcv['open time'].diff())),
    open=ohlcv['open'].to_numpy(),
    high=ohlcv['high'].to_numpy(),
    low=ohlcv['low'].to_numpy(),
    close=ohlcv['close'].to_numpy()
)

ohlcv_dict['open time'] = ohlcv['open time'].to_numpy()

symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']


# ======================================================================================
def change_atr_period(new_atr_period: int):
    raise NotImplementedError("should be able to change the ATR period!")


@execution_time
def get_max_leverage():
    return pm.max_leverage(data=ohlcv_dict, risk_level=3)


@execution_time
def get_diversifcation_multiplier(close_prices: np.ndarray):
    return pm.diversification_multiplier(close_prices=close_prices)


# --------------------------------------------------------------------------------------
@execution_time
def test_vol_anno(close: np.ndarray):
    ohlcv_dict['volatility'] = pm.vol_anno(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
        use_log_returns=False
    )


@execution_time
def test_vol_anno_nb(close: np.ndarray):
    ohlcv_dict['volatility'] = pm.vol_anno_nb(
        close=close,
        interval_in_ms=90_000,
        lookback=14,
    )


@execution_time
def test_conservative_sizing(data: dict):
    pm._conservative_sizing(data=data, target_risk_annual=0.1, smoothing=1)


@execution_time
def test_aggressive_sizing(data: dict):
    pm._aggressive_sizing(data=data, risk_limit_per_trade=0.05)


@execution_time
def test_max_leverage_fast(data: dict):
    ohlcv_dict['leverage'] = pm.max_leverage(data=data, risk_level=1)


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
    #     test_vol_anno_nb(ohlcv_dict['close'])

    # for _ in range(5):
    #     test_vol_anno(ohlcv_dict['close'])

    # for _ in range(5):
    #     test_conservative_sizing(ohlcv_dict)

    # for _ in range(5):
    #     test_aggressive_sizing(ohlcv_dict)

    for _ in range(5):
        test_max_leverage_fast(ohlcv_dict)

    # df = pd.DataFrame.from_dict(ohlcv_dict)
    # print(df.tail(10))

    test_get_diversification_multiplier(60, 5)
