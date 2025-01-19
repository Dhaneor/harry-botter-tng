#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import time
from pprint import pprint
import numpy as np
from cProfile import Profile  # noqa: E402, F401
import pstats  # noqa: E402, F401
import logging

import analysis.indicators.indicator as indicator  # noqa: E402
from analysis.chart.plot_definition import (  # noqa: E402, F401
    SubPlot, Line, Channel, Histogram
)
from util import get_logger

logger = get_logger("main", level="DEBUG")


array_len = 1000

a = np.random.rand(array_len) * 100
a1 = np.random.rand(array_len) * 1.05
a2 = np.random.rand(array_len) * 0.95

# ==============================================================================
# define some indicators and their SubPlots (plot descriptions) with
# examples that have/test for: 
# • one or multiple outputs
# • line elements
# • channel elements
# • histogram elements

defs = {
    "SMA": {
        "src": "talib",
        "params": {"timeperiod": 100},
        "subplots": SubPlot(
            label="Simple Moving Average (100)",
            is_subplot=False,
            elements=[
                Line(
                    label="SMA (100)", 
                    column="sma_100_close",
                    legend="SMA (100)",
                    legendgroup="SMA",
                    )
            ],
            level="indicator"
        )
    },
    "STOCH": {
        "src": "talib",
        "params": {
            "fastk_period": 14,
            "slowk_period": 10,
            "slowd_period": 10,
            "slowk_matype": 0,
            "slowd_matype": 0,
        },
        "subplots": SubPlot(
            label="Stochastic (14,10,0,10,0)",
            is_subplot=True,
            elements=[
                Line(
                    label="SLOWK", 
                    column="stoch_14_10_0_10_0_high_low_close_slowk",
                    legend="SLOWK",
                    legendgroup="STOCH"
                ),
                Line(
                    label="SLOWD", 
                    column="stoch_14_10_0_10_0_high_low_close_slowd",
                    legend="SLOWD",
                    legendgroup="STOCH",
                ),
            ],
            level="indicator"
        )
    },
    "BBANDS": {
        "src": "talib",
        "params": {
            "timeperiod": 20,
            "nbdevup": 2.5,
            "nbdevdn": 2.5,
            "matype": 0,
        },
        "subplots": SubPlot(
            label="Bollinger Bands (20,2.5,2.5,0)",
            is_subplot=False,
            elements=[
                Line(
                    label="MIDDLEBAND", 
                    column="bbands_20_2.5_2.5_0_close_middleband",
                    legend="MIDDLEBAND",
                    legendgroup="BBANDS"
                ),
                Channel(
                    label="BBANDS",
                    upper=Line
                    (
                        label="UPPERBAND", 
                        column="bbands_20_2.5_2.5_0_close_upperband",
                        legend="UPPERBAND",
                        legendgroup="BBANDS"
                    ),
                    lower=Line(
                        label="LOWERBAND", 
                        column="bbands_20_2.5_2.5_0_close_lowerband",
                        legend="LOWERBAND",
                        legendgroup="BBANDS"
                    )
                )
            ],
            level="indicator",
        )
    },
    "MACD": {
        "src": "talib",
        "params": {
            "fastperiod": 12,
            "slowperiod": 26,
            "signalperiod": 9
        },
        "subplots": SubPlot(
            label="Moving Average Convergence/Divergence (12,26,9)",
            is_subplot=True,
            elements=[
                Line(
                    label="MACD", 
                    column="macd_12_26_9_close_macd",
                    legend="MACD",
                    legendgroup="MACD"
                ),
                Line(
                    label="MACDSIGNAL", 
                    column="macd_12_26_9_close_macdsignal",
                    legend="MACDSIGNAL",
                    legendgroup="MACD"
                ),
                Histogram(
                    label="MACDHIST", 
                    column="macd_12_26_9_close_macdhist",
                    legend="MACDHIST",
                    legendgroup="MACD"
                )
            ],
            level="indicator"
        )
    },
#     "RSI_OVERBOUGHT": {
#         "src": "fixed",
#         "params": {"value": 70},
#         "parameter_space": {"trigger": [70, 100]},
#         # "subplots": SubPlot(
#         #     label="rsi_overbought_70",
#         #     is_subplot=True,
#         #     elements=[],
#         #     triggers=[("rsi_overbought_70", "Line")],
#         #     channel=[],
#         #     hist=[],
#         #     level="indicator"
#         # )
#     },
#     "RSI_OVERSOLD": {
#         "src": "fixed",
#         "params": {"value": 30},
#         "parameter_space": {"trigger": [0, 30]},
#         # "subplots": SubPlot(
#         #     label="rsi_oversold_30",
#         #     is_subplot=True,
#         #     elements=[],
#         #     triggers=[("rsi_oversold_30", "Line")],
#         #     channel=[],
#         #     hist=[],
#         #     level="indicator"
#         # )
#     },
}


# ==============================================================================

def test_indicator_factory(
    name: str,
    params: dict | None = None,
    src: str | None = None,
    show: bool = False
) -> indicator.Indicator:

    ind = indicator.factory(name, params, src)

    if show:
        print(ind)
        # ind.help()
        print("-------------------------")
        print("name:", ind.name)
        print("input:", ind.input)
        print("params:", ind.parameters)
        print("out:", ind.output)
        print("unique name:", ind.unique_name)

    assert isinstance(ind, indicator.IIndicator)
    assert ind.input is not None
    assert ind.parameters is not None
    assert ind.output is not None
    return ind


def test_set_indicator_parameters():
    for cand in defs:
        i = indicator.factory(
            indicator_name=cand,
            params=defs[cand]["params"],
            source=defs[cand]["src"],
        )

        if params := defs[cand].get("params"):
            for k, v in params.items():
                logger.info("setting parameter %s for %s -> %s", k, i, v * 2)
                i.parameters = {k: v * 2}
                assert i.parameters_dict[k] == v * 2

        logger.debug("   indicator: %s" % i)
        logger.debug("   unique name: %s" % i.unique_name)
        logger.debug("   unique output: %s" % i.unique_output)
        break


def test_indicator_run(i: indicator.IIndicator, params: dict | None = None):

    if params:
        i.parameters = params

    match len(i.input):
        case 1:
            res = i.run(a)
        case 2:
            res = i.run(a, a1)
        case 3:
            res = i.run(a, a1, a2)
        case 4:
            res = i.run(a, a1, a2, a2 * 1.1)
        case _:
            raise ValueError(f"invalid input: {i.input}")

    assert res is not None

    if isinstance(res, tuple):
        assert len(res) == len(i.output)
        assert res[0].shape[0] == a.shape[0]
    else:
        assert res.shape[0] == a.shape[0]

    return res


def test_unique_name(ind):
    name = ind.unique_name
    assert isinstance(name, str)
    assert len(name) > 0
    # assert name.startswith(ind.__class__.__name__)
    print(ind.__class__.__name__)
    return name


def test_subplots():
    for cand in defs:
        logger.info("=" * 200)
        i = indicator.factory(cand, defs[cand]["params"], defs[cand]["src"])

        result = i.subplots
        exp = defs[cand]["subplots"].__dict__
        res = result[0].__dict__
        
        for k in exp.keys():
            
            try:
                assert res[k] == exp[k]
            except AssertionError as e:
                logger.error(f"[{k}] ===== EXPECTED {exp[k]}")
                logger.error(f"[{k}] ===== GOT: {res[k]}")
                logger.error("~" * 200)

                if k == "elements":
                    exp_elems = exp[k]
                    res_elems = res[k]

                    for i in range(len(exp_elems)):
                        try:
                            assert exp_elems[i] == res_elems[i]
                        except AssertionError:
                            for k, v in exp_elems[i].__dict__.items():
                                res_v = res_elems[i].__dict__[k]

                                if not v == res_v:
                                    logger.info("%s: %s ----- %s",  k, v, res_v)

                return
        else:
            logger.info(i.subplots)
    else:
        logger.info("verify plot descriptions: OK")


def test_plot(indicator: indicator.Indicator):
    """Tests the plot method of the Indicator class"""
    inputs = len(indicator.input)

    print(indicator)

    match inputs:
        case 1:
            indicator.run(np.random.rand(100).reshape(-1, 1) * 10_000)
        case 2:
            indicator.run(
                np.random.rand(100).reshape(-1, 1) * 10_000, 
                np.random.rand(100).reshape(-1, 1) * 10_000
            )

    try:
        indicator.plot()
    except AttributeError as e:
        logger.error(e)
        logger.error(dir(indicator))
    except Exception as e:
        logger.error(f"Error plotting indicator: {indicator.__class__.__name__}")
        logger.error(f"{str(e)}", exc_info=True)



# tests for the Parameter class
def test_parameter():
    """Tests the Parameter class"""
    p = indicator.Parameter(
        name="slowperiod",
        _value=1,
        min_=0.1,
        max_=10,
        hard_min=0,
        hard_max=10,
    )

    logger.info(p.__dict__)

    for v in (1, 3, 5, 8.78, 10, -1, 11, "this_string", [5, 10, 12.8]):
        try:
            p.value = v

            if p._enforce_int:
                assert p.value == int(
                    round(v)
                ), f"expected {int(round(v))}, but got {p.value}"
            else:
                assert p.value == v, f"expected {v}, but got {p.value}"
            logger.info("parameter value %s: OK (%s)", v, p.value)

        except ValueError as e:
            success = "OK"
            logger.info("parameter value %s out of range: %s (%s)", v, success, e)

        except TypeError as e:
            success = "OK"
            logger.info("parameter value %s has wrong type: %s (%s)", v, success, e)


def test_parameter_space():
    """Tests the setting the parameter space for a Parameter class"""
    p = indicator.Parameter(
        name="slowperiod",
        _value=1,
        min_=0.1,
        max_=10,
        hard_min=0,
        hard_max=10,
    )

    print(p.__dict__)

    logger.info("parameter space: %s", p.space)
    logger.info("hard minimum: %s / hard maximum: %s", p.hard_min, p.hard_max)

    up = (round(x * 0.12, 2) for x in range(-40, 120, 5))
    down = (round(x * 0.12, 2) for x in range(130, 30, -5))

    for min_, max_ in zip(up, down):
        try:
            p.space = min_, max_
            logger.info(
                "set parameter space to [%s, %s] for requested values [%s, %s]: OK",
                p.min_,
                p.max_,
                min_,
                max_,
            )
        except ValueError as e:
            logger.error("set parameter space to [%s, %s]: FAIL -> %s", min_, max_, e)

    logger.info("finished: OK")


def test_parameter_iter():
    p = indicator.Parameter(
        name="test",
        _value=1,
        min_=0,
        max_=10,
        hard_min=0,
        hard_max=10,
    )

    pprint(p.__dict__)

    logger.info("parameter space for %s: %s", p, p.space)
    for elem in p:
        logger.info(elem)


def test_run_with_mutiple_parameters(ind):
    logger.info(ind.parameters)
    logger.info(
        "unique name: %s, output name(s): %s",
        ind.unique_name, ind.unique_output
        )
    first = test_indicator_run(ind)
    logger.info(first[-5:])

    for p in ind.parameters:
        p.increase()

    logger.info(ind.parameters)
    logger.info(
        "unique name: %s, output name(s): %s",
        ind.unique_name, ind.unique_output
        )
    second = test_indicator_run(ind)
    logger.info(second[-5:])

    assert list(first) != list(second), "same result for different parameters"


def test_randomize(ind):
    logger.info(ind.parameters)
    logger.info(ind.parameter_space)

    for _ in range(10):
        logger.info("-" * 80)

        try:
            ind.randomize()
        except Exception as e:
            logger.error("randomize failed: %s", e, exc_info=True)
            return

        logger.info(ind.parameters)


def test_return_type():
    ind = indicator.factory("ER", {}, None)

    assert isinstance(ind, indicator.IIndicator)

    shape = 30, 2
    data = np.random.rand(*shape)

    logger.debug(f"data dimensions: {data.ndim} shape: {data.shape}")

    res = ind.run(data)

    logger.info(res)

    # while isinstance(res, (np.ndarray, list, tuple)):
    #     logger.debug(type(res))
    #     res = res[0]

    assert isinstance(res[0], np.ndarray), \
        f"Expected np.ndarray, got {type(res)}"
    
    assert res[0].shape == shape, f"Expected shape {shape}, got {res.shape}"



# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    # test_indicator_factory("STOCH", show=True)
    # test_set_indicator_parameters()
    # test_randomize(test_indicator_factory("BBANDS"))
    # test_return_type()
    # test_parameter_space()
    # test_parameter_iter()
    # test_subplots()
    # sys.exit()

    # arr, res = test_is_above()

    # ind = test_indicator_factory(
    #     "BBANDS", {"timeperiod": 80, "nbdevup": 1.5}, show=False
    # )

    # ind = test_indicator_factory(
    #     "LINEARREG", params={"timeperiod": 20}, show=False
    #     )

    # ind = test_indicator_factory(
    #     "ER", params={"timeperiod": 30}, show=False
    #     )

    ind = test_indicator_factory("AROONOSC", show=False)
    test_plot(ind)

    # ind = test_indicator_factory("STOCH", show=False)

    # ind = test_indicator_factory(
    #     "rsi_oversold",
    #     {"value": 70, "parameter_space": {"trigger": [30, 70]}},
    #     "fixed",
    #     show=False
    # )

    # ind.parameters = {"value": 80, "parameter_space": [40, 70]}

    # print(ind.help())
    # pprint(ind.subplots)
    # pprint(ind.__dict__)
    # print("unqiue_output: ", ind.unique_output)
    # print("unique_name: ", ind.unique_name)
    # print(ind.subplots)

    # test_run_with_mutiple_parameters(ind)

    sys.exit(0)

    runs = 1_000_000
    logger.setLevel(logging.ERROR)
    st = time.time()

    # for _ in range(runs):
    #     ind = test_indicator_factory(
    #         "BBANDS", {"timeperiod": 80, "nbdevup": 1.5},
    #         show=False
    #     )
    #     ind.parameter_space = {"value": [40, 70]}

    with Profile() as p:
        for _ in range(runs):
            ind.parameter_space = {"value": [40, 70]}

        (
            pstats.Stats(p)
            .strip_dirs()
            .sort_stats(pstats.SortKey.CUMULATIVE)
            # .reverse_order()
            .print_stats(20)
        )

    print(f"execution time: {((time.time() - st)*1_000_000/runs):.2f} microseconds")
