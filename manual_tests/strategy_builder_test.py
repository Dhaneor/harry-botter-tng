#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import time
import logging
import numpy as np

from pprint import pprint
from random import choice, random

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

from analysis import strategy_builder as sb, MarketData
from analysis.strategy import exit_order_strategies as es
from analysis.strategy.definitions import (  # noqa: F401
    tema_cross, breakout, rsi,
    s_test_er, s_tema_cross, s_linreg, s_trix, s_breakout, s_kama_cross,
)
from util import get_logger

logger = get_logger("main")

data = MarketData.from_random(length=2000, no_of_symbols=1)

# -----------------------------------------------------------------------------
def __get_sl_strategy_definition():
    return es.StopLossDefinition(
        strategy="atr",
        params=dict(
            atr_lookback=14,
            atr_factor=3,
            is_trailing=True,
        ),
    )


def __get_tp_strategy_definition():
    return es.TakeProfitDefinition(
        strategy="atr",
        params=dict(
            atr_lookback=14,
            atr_factor=5,
            is_trailing=True,
        ),
    )


def __get_single_strategy_definition():
    return sb.StrategyDefinition(
        strategy="TEMA CROSS",
        symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
        interval="1d",
        signals_definition=tema_cross,
        weight=1,  # random(),
    )


def __get_composite_strategy_definition():
    return sb.StrategyDefinition(
        strategy="Composite Strategy",
        symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
        interval="1d",
        sub_strategies=[
            sb.StrategyDefinition(
                strategy="CCI",
                symbol="BTCUSDT",
                interval="1d",
                signals_definition=breakout,
                weight=0.7,
            ),
            sb.StrategyDefinition(
                strategy="RSI",
                symbol="BTCUSDT",
                interval="1d",
                signals_definition=rsi,
                weight=0.3,
            ),
        ]
    )


# ..............................................................................
def build_valid_single_strategy():
    sdef = __get_single_strategy_definition()

    try:
        s = sb.build_strategy(sdef)
    except Exception as e:
        logger.exception(e)
        return None

    assert isinstance(s, sb.IStrategy)
    assert s.symbol is not None
    assert s.interval is not None
    assert s.weight is not None

    return s


def build_valid_composite_strategy(
    symbol="BTCUSDT",
    interval="1d",
):
    strategy_definition = {
        "name": "Base Strategy Composite",
        "symbol": symbol,
        "interval": interval,
        "sub_strategies": {
            "Base Strategy Single 1": {
                "name": "Base Strategy Single",
                "symbol": "BTCUSDT",
                "interval": interval,
                "test": False,
                choice(("wrong_param", "not_exist", "holy crap")): None,
            },
            "Base Strategy Single 2": {
                "name": "Base Strategy Single",
                "symbol": "BTCUSDT",
                "interval": interval,
                "weight": 2,
                "test": False,
                choice(("wrong_param", "not_exist", "holy crap")): None,
            },
        },
    }

    sl_params = {
        "strategy": "atr",
        "atr_lookback": 14,
        "atr_factor": 5,
        "is_trailing": True,
    }

    tp_params = {
        "strategy": "atr",
        "atr_lookback": 10,
        "atr_factor": 6,
        "is_trailing": False,
        # 'tp crap:': choice(('whatever', 'never mind', 'holy crap'))
    }

    s = sb.build_strategy(
        strategy_definition=strategy_definition,
        sl_params=sl_params,
        tp_params=tp_params,
    )

    print("\n", s, "\n\n")


# ..............................................................................
def test_get_strategy_definition():
    pprint(__get_single_strategy_definition())


# @execution_time
def test_strategy_builder():
    return build_valid_single_strategy()


def test_strategy_run(s):
    res = s.speak()

    assert isinstance(res, np.ndarray)
    assert res.ndim == 3

    print(res)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    # close = df.close.to_numpy()[-100_000:]
    # close = np.random.rand(1_000)
    # print(close.shape)

    if s := test_strategy_builder():
        s.market_data = data
        print('-' * 200)
        print(s)

    # test_sl_strategy_factory()
    # sdef = __get_single_strategy_definition()

    # test_serialize_object(s)

    test_strategy_run(s)

    # ..........................................................................
    # sys.exit()

    logger.setLevel(logging.ERROR)
    runs = 100_000
    data = data
    st = time.perf_counter()

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            s.speak()

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    # for _ in range(runs):
    #     test_strategy_run(s, False)

    et = time.perf_counter() - st
    ips = runs / et
    periods = len(data["close"]) * ips

    print(f'data: {len(data["close"]):,} periods')
    print(f"periods/s: {periods:,.0f}")
    print(f"\navg exc time: {(et * 1_000_000 / runs):.0f} µs")

    print(f"\n~iter/s (1 core): {ips:>10,.0f}")
    print(f"~iter/s (8 core): {ips * 5:>10,.0f}")
    print(f"~iter/m (8 core): {ips * 5 * 60:>10,.0f}")
