#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import time
import logging
import numpy as np
import pandas as pd

from pprint import pprint
from random import choice, random, randint

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

# configure logger
LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("main")
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

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

from src.analysis import strategy_builder as sb  # noqa: E402
from src.analysis.strategies import operand as op  # noqa: E402, F401
from src.analysis.strategies import condition as cnd  # noqa: E402, F401
from src.analysis.strategies import signal_generator as sg  # noqa: E402, F401
from src.analysis.strategies import exit_order_strategies as es  # noqa: E402
from src.analysis.strategies.definitions import (  # noqa: E402, F401
    cci, rsi, ema_cross, tema_cross
)  # noqa: E402, F401


df = pd.read_csv(os.path.join(parent, "ohlcv_data", "btcusdt_15m.csv"))
df.drop(
    ["Unnamed: 0", "close time", "volume", "quote asset volume"], axis=1, inplace=True
)
length = 1500
start = randint(0, len(df) - length)
end = start + length
# df = df[start: end]

data = {col: df[start:end][col].to_numpy() for col in df.columns}


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
        weight=random(),
        stop_loss=None,  # [__get_sl_strategy_definition() for _ in range(1)],
        take_profit=None,  # [__get_tp_strategy_definition() for _ in range(1)],
        params={
            "timeperiod": 28,
            # 'test': False,
            # 'wrong_param': None,
            # 'not_exist': None,
            # 'holy_crap': None
        },
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
                signals_definition=cci,
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

    assert isinstance(s.sl_strategy[0], es.IStopLossStrategy)
    assert isinstance(s.tp_strategy[0], es.ITakeProfitStrategy)

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
    # build_valid_composite_strategy()


def test_strategy_run(s, show=False):

    s.speak(data)

    if show:
        df = pd.DataFrame.from_dict(data)

        df.replace(False, ".", inplace=True)
        df.replace(np.nan, ".", inplace=True)
        df.replace(0.0, ".", inplace=True)

        print(df.info())
        print(sys.getsizeof(data))
        print(df.tail(50))


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    # close = df.close.to_numpy()[-100_000:]
    # close = np.random.rand(1_000)
    # print(close.shape)

    # if s := test_strategy_builder():
    #     print('-' * 200)
    #     print(s)

    # test_sl_strategy_factory()
    # sdef = __get_single_strategy_definition()
    sdef = __get_composite_strategy_definition()

    print(sdef)

    s = sb.build_strategy(sdef)
    assert isinstance(s, sb.IStrategy)

    print("-" * 200)
    print(s)

    test_strategy_run(s, True)

    # ..........................................................................
    sys.exit()

    logger.setLevel(logging.ERROR)
    runs = 1_000
    data = data
    st = time.perf_counter()

    for i in range(runs):
        test_strategy_run(s, False)

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            s.speak(data)

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
    print(f'length data: {len(data["close"])} periods')
    print(f"execution time: {((et - st)*1_000_000/runs):.2f} microseconds")
