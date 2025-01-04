#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import time
import logging
import numpy as np
import pandas as pd
import sys

from pprint import pprint  # noqa: F401, F501

# profiler imports
from cProfile import Profile  # noqa F401
from pstats import SortKey, Stats  # noqa: F401

from analysis.strategy import operand as op
from analysis.strategy.operand_factory import operand_factory
from analysis.chart.plot_definition import SubPlot
from util import get_logger

# configure logger
logger = get_logger('main', level=logging.DEBUG)


# ======================================================================================
columns = ["open time", "open", "high", "low", "close", "volume"]
data = {col: np.random.rand(1_000) for col in columns}

# ======================================================================================
# define different operands for testing possible variants
op_defs = {
    "sma": {
        "def": ('sma', {'timeperiod': 20},),
        "params": {"timeperiod": 30},
        # "plot_desc": SubPlot(
        #     label='Simple Moving Average (20)',
        #     is_subplot=False,
        #     elements=[Line(label="sma_20", column="sma_20", end_marker=False)],
        #     level='operand'
        # )
    },

    "kama": {
        "def": ('kama', {'timeperiod': 20}),
        "params": {"timeperiod": 30},
        # "plot_desc": SubPlot(
        #     label='Kaufman Adaptive Moving Average (20)',
        #     is_subplot=False,
        #     lines=[('kama_20', 'Line')],
        #     triggers=[],
        #     channel=[],
        #     level='operand'
        # )
    },

    "bbands": {
        "def": ('BBANDS.upperband', {'timeperiod': 10},),
        "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
        # "plot_desc": SubPlot(
        #     label='Bollinger Bands (10 2 2 0)',
        #     is_subplot=False,
        #     lines=[('bbands_10_2_2_0_middleband', 'Line')],
        #     triggers=[],
        #     channel=[
        #         'bbands_10_2_2_0_upperband',
        #         'bbands_10_2_2_0_lowerband'
        #     ],
        #     level='operand'
        # )
    },

    "rsi": {
        "def": ('rsi', {'timeperiod': 14},),
        "params": {"timeperiod": 28},
        # "plot_desc": SubPlot(
        #     label='Relative Strength Index (14)',
        #     is_subplot=True,
        #     lines=[('rsi_14', 'Line')],
        #     triggers=[],
        #     channel=[],
        #     level='operand'
        # )
    },

    "macd": {
        "def": ('macd.macdsignal', {'fastperiod': 3, 'slowperiod': 15}),
        "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        # "plot_desc": SubPlot(
        #     label='Moving Average Convergence/Divergence (9 26 9)',
        #     is_subplot=True,
        #     lines=[
        #         ('macd_9_26_9_macd', 'Line'),
        #         ('macd_9_26_9_macdsignal', 'Dashed Line')
        #     ],
        #     triggers=[],
        #     channel=[],
        #     hist=['macd_9_26_9_macdhist'],
        #     level='operand'
        # )
    },

    'close': {
        "def": 'close',
        "params": None,
        #     "plot_desc": SubPlot(
        #         label='close',
        #         is_subplot=False,
        #         lines=[],
        #         triggers=[],
        #         channel=[],
        #         level='operand'
        #     )
    },

    'rsi_overbought': {
        "def": ('rsi_overbought', 80.5, [70, 100]),
        "params": {"trigger": 90},
        'parameter_space': {'rsi_overbought': [70, 100]},
        # "plot_desc": SubPlot(
        #     label='Rsi Overbought 80.5',
        #     is_subplot=True,
        #     elements=[
        #         Line(
        #             label="rsi_overbought_80.5",
        #             column="rsi_overbought_80.5",
        #             end_marker=False)
        #         ],
        #     level='operand'
        # )
    },

    'sma_of_rsi': {
        "def": ("sma", ("rsi", {"timeperiod": 7}), {"timeperiod": 28},),
        "params": {"timeperiod": 45},
        # "plot_desc": SubPlot(
        #     label='Simple Moving Average (28) of RSI (7)',
        #     is_subplot=True,
        #     lines=[('sma_28_rsi_7', 'Line')],
        #     triggers=[],
        #     channel=[],
        #     level='operand'
        # )
    },
    'er': {
        'def': ("er", {"timeperiod": 7})
    }
}


# =====================================================================================
def update_parameters_callback(*args):
    logger.debug(f'notification about parameter update: {args}')


def test_operand_factory():
    for elem in op_defs:
        op_def = op_defs.get(elem).get("def")
        operand = operand_factory(op_def)

        logger.debug("[TEST] %s" % operand)
        logger.debug('[TEST] ======================================================')
        # assert isinstance(operand, op.Operand), \
        #     (
        #         "expected resutl for %s to be an instance of 'Operand', but got %s"
        #         % (elem, type(operand))
        #     )
        if operand:
            pprint(operand.__dict__)
        if elem == "rsi_overbought":
            break

    return operand


def test_nested_indicators():
    op_def = ("sma", ("rsi", {"timeperiod": 16}), {"timeperiod": 48},)
    operand = operand_factory(op_def)
    operand.randomize()
    operand.randomize()

    logger.debug("[TEST] %s" % operand)
    logger.debug('[TEST] ======================================================')
    assert isinstance(operand, op.Operand), \
        (
            "expected resutl for %s to be an instance of 'Operand', but got %s"
            % (op_def, type(operand))
        )
    if operand:
        pprint(operand.__dict__)

def test_operand_run(operand: op.Operand, data: dict, show_result: bool = False):
    logger.debug(operand)
    res = operand.run(data)
    # assert isinstance(res, dict)
    # assert len(res) == 1
    # assert res['sma'] is not None
    # assert isinstance(res['sma'], np.ndarray)
    # assert res['sma'].shape == (10,)
    # assert res['sma'].dtype == np.float64
    # assert res['sma'].min() > 0

    if res is not None and show_result:
        print("-~•~-" * 40)
        df_res = pd.DataFrame.from_dict(res)
        print(df_res.tail(10))


def test_plot_desc():
    logger.setLevel(logging.INFO)

    for elem in op_defs:
        op_def = op_defs.get(elem).get("def")
        operand = op.operand_factory(op_def)

        logger.info('---------------------------------------------------------')
        logger.info(operand)
        logger.info(' ')
        for i in operand.indicators:
            logger.info("\t\t%s", i.plot_desc)
        logger.info(operand.plot_desc)

        # expected = op_defs.get(elem).get("plot_desc")

        assert isinstance(operand.plot_desc, SubPlot)
        # assert expected == operand.plot_desc, f"{expected} != {operand.plot_desc}"

    logger.info('---------------------------------------------------------')
    logger.info("plot description validation: OK")


def test_update_parameters():

    for elem in list(op_defs.keys()):
        if elem == 'rsi_overbought':
            try:
                logger.debug('=' * 150)
                op_def = op_defs.get(elem).get("def")
                operand = op.operand_factory(op_def)

                if not hasattr(operand, 'indicators'):
                    if isinstance(operand, op.OperandPriceSeries):
                        logger.warning(f"Operand {elem} has no indicators")
                        continue
                    else:
                        raise AttributeError(f"Operand {elem} has no indicators")

                for indicator in operand.indicators:
                    logger.debug('-' * 150)
                    logger.debug("before: %s" % indicator)
                    indicator.randomize()
                    logger.debug("after: %s" % indicator)

                logger.debug("before: %s" % op.operand_factory(op_def))
                logger.debug("after: %s" % operand)
            except Exception as e:
                logger.error(f"Error updating parameters for {elem}: {e}")
                continue

            pprint(operand.__dict__)
            pprint(operand.indicators[0].__dict__)

    logger.info("update parameters: OK")


def test_randomize():
    op_def = op_defs.get('sma').get("def")
    operand = operand_factory(op_def)

    logger.info("before: %s" % operand)

    operand.randomize()
    operand.randomize()

    logger.info("-" * 150)
    logger.info("after: %s" % operand)
    pprint(operand.__dict__)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    # test_operand_factory()
    # test_nested_indicators()
    # test_update_parameters()
    # test_randomize()
    # sys.exit()

    # operand = operand_factory(op_defs.get('bbands').get('def'))
    operand = operand_factory("sma")

    pprint(operand.__dict__)

    for _ in range(2):
        logger.debug('=' * 120)
        operand.run(data)
        logger.debug(list(data.keys()))

    # print(pd.DataFrame.from_dict(data).tail(10))

    # sys.exit()

    # print('-~•~-' * 40)
    # print("indicator:")
    # pprint(operand.indicator.plot_desc)
    # print('-~•~-' * 40)
    # print("operand:")
    # pprint(operand.as_dict())
    # print(operand.indicator.display_name)
    # print('-~•~-' * 40)
    # print("plot description:")
    # pprint(operand.plot_desc)

    sys.exit()

    logger.setLevel(logging.ERROR)
    runs = 10_000
    data = data
    st = time.time()

    op_def = op_defs.get('sma').get('def')
    operand = op.operand_factory(op_def)

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = operand.randomize()
            # op.update_parameters({op.unique_name: {'timeperiod': i}})

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)
    )

    # for _ in range(runs):
    #     test_execute_condition(data)

    print(f'length data: {len(data["close"])} periods')
    print(
        f"avg execution time: {(((time.time() - st) * 1_000_000) / runs):.2f} microseconds"
    )

    # pprint(operand.as_dict())
