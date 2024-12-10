#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import time
from functools import reduce
import itertools as it
import logging
import numpy as np
import pandas as pd
from typing import Iterable
from pprint import pprint  # noqa: F401
from random import choice

# profiler imports
from cProfile import Profile  # noqa: F401
from pstats import SortKey, Stats  # noqa: F401

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from src.analysis.strategies import condition as cn  # noqa: E402
from src.analysis.strategies import operand as op  # noqa: E402
from helpers_ import get_sample_data  # noqa: E402

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# ======================================================================================
interval = "4h"
length = 1500
data = get_sample_data(length)

# ======================================================================================
c_defs = {
    "ema": cn.ConditionDefinition(
        interval="15m",
        operand_a=("ema", {"timeperiod": 63}),
        open_long="close is above a",
        open_short="close is below a",
    ),
    "golden_cross": cn.ConditionDefinition(
        interval="15m",
        operand_a=("sma", {"timeperiod": 50}),
        operand_b=("sma", {"timeperiod": 200}),
        open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
        open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
    ),
    "rsi": cn.ConditionDefinition(
        interval="15m",
        operand_a=("rsi", {"timeperiod": 10}),
        operand_b=("rsi_os", 30),
        operand_c=("rsi_ob", 70),
        operand_d=("rsi_50", 50),
        open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
        open_short=("a", cn.COMPARISON.CROSSED_BELOW, "c"),
        close_long=("a", cn.COMPARISON.CROSSED_ABOVE, "d"),
        close_short=("a", cn.COMPARISON.CROSSED_BELOW, "d"),
    ),
    "macd": cn.ConditionDefinition(
        interval="15m",
        operand_a=("macd", {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
        open_long=("a", cn.COMPARISON.IS_ABOVE, 0),
        open_short=("a", cn.COMPARISON.IS_BELOW, 0),
    ),
    "bbands": cn.ConditionDefinition(
        interval="15m",
        operand_a=("bbands", {"timeperiod": 20, "matype": 0}),
        open_long=("close", cn.COMPARISON.IS_ABOVE, "a.upperband"),
        open_short=("close", cn.COMPARISON.IS_BELOW, "a.lowerband"),
    ),
}


# ======================================================================================
def test_condition_definition():
    for name in c_defs.keys():
        d = c_defs[name]
        logger.debug(d)

        assert isinstance(d, cn.ConditionDefinition)
        assert isinstance(d.interval, str)


def test_condition_factory():
    factory = cn.ConditionFactory()

    for name, c_def in c_defs.items():
        try:
            condition = factory.build_condition(c_def)
        except Exception as e:
            logger.exception(e)
            break

        pprint(condition.__dict__)

        assert isinstance(condition.operand_a, op.Operand), \
            f"operand A of {name} is != instance of Operand"

        if condition.operand_b is not None:
            assert isinstance(condition.operand_b, op.Operand), \
                f"operand b of {name} is != instance of Operand"

        if condition.operand_c is not None:
            assert isinstance(condition.operand_c, op.Operand), \
                f"operand c of {name} is != instance of Operand"

        at_least_one_trigger = False

        for trig in ('open_long', 'open_short', 'close_long', 'close_short'):
            if getattr(condition, trig) is not None:
                at_least_one_trigger = True
                assert isinstance(getattr(condition, trig), tuple), \
                    f"trigger {trig} of {name} is != instance of tuple"
                assert len(getattr(condition, trig)) == 3, \
                    f"trigger {trig} of {name} is != length 3"

        assert at_least_one_trigger
        logger.info(condition)


def test_condition_is_working(c: cn.Condition):
    it_does = c.is_working()
    assert it_does
    logger.debug("condition is working: %s", it_does)
    return c


def test_execute_condition(
    data: dict,
    condition: cn.Condition,
    show_result: bool = False
):
    try:
        res = condition.execute(data)
    except Exception as e:
        logger.exception(e)
        res = None

    if res is not None and show_result:
        data.update(res.as_dict())
        df_res = pd.DataFrame.from_dict(data)
        df_res.set_index(keys=['human open time'], inplace=True)
        df_res.drop('volume', axis=1, inplace=True)
        df_res.replace(False, '•', inplace=True)
        df_res.replace(np.nan, '', inplace=True)
        print('\n' + '~_*_~' * 25)
        print(df_res.tail(50))


def test_condition_indicators(condition: cn.Condition):
    indicators = condition.indicators

    assert isinstance(indicators, Iterable), "Indicators must be iterable"
    assert all(
        isinstance(elem, op.ind.IIndicator) for elem in indicators
    ), "All indicators must be instances of Indicator"

    return indicators


def test_condition_result():
    def get_random_array(length=10):
        return np.array(tuple(choice([True, False]) for _ in range(length)))

    def get_random_condition_result():
        return cn.ConditionResult(
            open_long=get_random_array(),
            close_long=get_random_array(),
            open_short=get_random_array(),
            close_short=get_random_array(),
        )

    cr1 = get_random_condition_result()
    cr2 = get_random_condition_result()
    cr3 = get_random_condition_result()

    res1 = tuple(it.accumulate([cr1, cr2, cr3], lambda x, y: x & y))[-1]
    res2 = reduce(lambda x, y: x + y, [cr1, cr2, cr3])

    # print(cr1)
    # print(cr2)
    # print(cr3)
    # print('\n')
    # # print(res1)
    # print(res2)

    print(cr1.open_long)
    print(cr1.close_long)
    print(cr1.open_short)
    print(cr1.close_short)
    print
    print('-' * 120)
    print(cr1.combined())


def test_condition_result_from_combined():
    s1 = np.array(tuple(choice([1., 0., -1., np.nan]) for _ in range(10)))
    s2 = np.array(tuple(choice([1., 0., -1., np.nan]) for _ in range(10)))

    combined = s1 + s2

    cr = cn.ConditionResult.from_combined(combined)

    print(combined)
    print('-' * 50)
    print(cr.open_long)
    print(cr.open_short)
    print(cr.close_long)
    print(cr.close_short)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    test_condition_result_from_combined()

    sys.exit()

    c = cn.factory(c_defs["golden_cross"])

    for comp in (c.operand_a, c.operand_b, c.operand_c, c.operand_d):
        if comp:
            print(comp.indicators)

    # print(c.indicators)

    # sys.exit()

    test_condition_is_working(c)
    logger.debug("~-*-~" * 30)
    logger.debug("condition: %s", c)
    logger.debug("indicators: %s", test_condition_indicators(c))

    pprint(c.__dict__)
    print('-' * 80)
    # pprint(c.operand_a.plot_desc)
    # pprint(c.operand_b.plot_desc)
    # pprint(c.operand_c.plot_desc)
    # pprint(c.operand_d.plot_desc)
    # print('-' * 80)
    # pprint(c.plot_desc)

    # print(c)
    # test_execute_condition(data, c, True)

    # --------------------------------------------------------------------------
    sys.exit()

    cdef = c_defs['ema']
    factory = cn.ConditionFactory()

    logger.setLevel(logging.ERROR)
    runs = 1_000_000
    data = data
    st = time.time()

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = c.execute(data)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    # for _ in range(runs):
    #     test_execute_condition(data, c, False)

    print(f'length data: {len(data["close"])} periods')
    print(f"execution time: {((time.time() - st)*1_000_000/runs):.2f} microseconds")
