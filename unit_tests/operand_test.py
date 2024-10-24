#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import random   # noqa F401
import sys
import os
import time
import logging
import pandas as pd

from pprint import pprint  # noqa: F401, F501

# profiler imports
from cProfile import Profile  # noqa F401
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

from src.analysis.strategies import operand as op  # noqa: E402
from src.analysis.indicators import indicator as ind  # noqa: E402

# ======================================================================================
# load and prepare OHLCV data for use in testing
df = pd.read_csv(os.path.join(parent, "ohlcv_data", "btcusdt_15m.csv"))
df.drop(
    ["Unnamed: 0", "close time", "volume", "quote asset volume"], axis=1, inplace=True
)
df = df[-1_000:]

data = {col: df[col].to_numpy() for col in df.columns}

# ======================================================================================
# define different operands for testing possible variants
op_defs = {
    "sma": {
        "def": ('sma', {'timeperiod': 20},),
        "params": {"timeperiod": 30},
        "plot_desc": ind.PlotDescription(
            label='Simple Moving Average (20)',
            is_subplot=False,
            lines=[('sma_20', 'Line')],
            triggers=[],
            channel=[],
            level='operand'
        )
    },

    "kama": {
        "def": ('kama', {'timeperiod': 20}),
        "params": {"timeperiod": 30},
        "plot_desc": ind.PlotDescription(
            label='Kaufman Adaptive Moving Average (20)',
            is_subplot=False,
            lines=[('kama_20', 'Line')],
            triggers=[],
            channel=[],
            level='operand'
        )
    },

    "bbands": {
        "def": ('BBANDS', {'timeperiod': 10},),
        "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
        "plot_desc": ind.PlotDescription(
            label='Bollinger Bands (10 2 2 0)',
            is_subplot=False,
            lines=[('bbands_10_2_2_0_middleband', 'Line')],
            triggers=[],
            channel=[
                'bbands_10_2_2_0_upperband',
                'bbands_10_2_2_0_lowerband'
            ],
            level='operand'
        )
    },

    "rsi": {
        "def": ('rsi', {'timeperiod': 14},),
        "params": {"timeperiod": 28},
        "plot_desc": ind.PlotDescription(
            label='Relative Strength Index (14)',
            is_subplot=True,
            lines=[('rsi_14', 'Line')],
            triggers=[],
            channel=[],
            level='operand'
        )
    },

    "macd": {
        "def": ('macd.macdsignal', {'fastperiod': 9, 'slowperiod': 26}),
        "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "plot_desc": ind.PlotDescription(
            label='Moving Average Convergence/Divergence (9 26 9)',
            is_subplot=True,
            lines=[
                ('macd_9_26_9_macd', 'Line'),
                ('macd_9_26_9_macdsignal', 'Dashed Line')
            ],
            triggers=[],
            channel=[],
            hist=['macd_9_26_9_macdhist'],
            level='operand'
        )
    },

    'close': {
        "def": 'close',
        "params": None,
        "plot_desc": ind.PlotDescription(
            label='close',
            is_subplot=False,
            lines=[],
            triggers=[],
            channel=[],
            level='operand'
        )
    },

    'rsi_overbought': {
        "def": ('rsi_overbought', 80.5),
        "params": {"trigger": 90},
        'parameter_space': {'trigger': [70, 100]},
        "plot_desc": ind.PlotDescription(
            label='Rsi Overbought 80.5',
            is_subplot=True,
            lines=[],
            triggers=[('rsi_overbought_80.5', 'Line')],
            channel=[],
            level='operand'
        )
    },

    'sma_of_rsi': {
        "def": ("sma", ("rsi", {"timeperiod": 7}), {"timeperiod": 28},),
        "params": {"timeperiod": 45},
        "plot_desc": ind.PlotDescription(
            label='Simple Moving Average (28) of RSI (7)',
            is_subplot=True,
            lines=[('sma_28_rsi_7', 'Line')],
            triggers=[],
            channel=[],
            level='operand'
        )
    }
}
# ==============================================================================


def test_operand_factory():
    for elem in op_defs:
        op_def = op_defs.get(elem).get("def")
        operand = op.operand_factory(op_def)

        logger.debug(operand)
        logger.debug('----------------------------------------------------------------')
        assert isinstance(operand, op.Operand)

    return operand


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

        expected = op_defs.get(elem).get("plot_desc")

        assert isinstance(operand.plot_desc, ind.PlotDescription)
        assert expected == operand.plot_desc, f"{expected} != {operand.plot_desc}"

    logger.info('---------------------------------------------------------')
    logger.info("plot description validation: OK")


def test_update_parameters():

    for elem in op_defs.keys():
        logger.debug('=' * 120)
        op_def = op_defs.get(elem).get("def")
        operand = op.operand_factory(op_def)

        if not hasattr(operand, 'indicators'):
            if isinstance(operand, op.OperandPriceSeries):
                logger.warning(f"Operand {elem} has no indicators")
                continue
            else:
                raise AttributeError(f"Operand {elem} has no indicators")

        indicators = operand.indicators

        # logger.debug('=' * 70)
        logger.debug(operand)
        logger.debug("operand indicators: %s", indicators)

        for indicator in indicators:
            logger.debug('-' * 120)
            logger.debug(indicator)
            logger.debug("\tindicator parameters: %s", indicator.parameters)
            logger.debug("\tindicator unique name: %s", indicator.unique_name)
            params = op_defs.get(elem).get("params")

            operand.update_parameters({indicator.unique_name: params})

            logger.debug("%s == %s", indicator.parameters, params)
            logger.debug(operand)

            for key in params.keys():
                assert indicator.parameters[key] == params[key]

    logger.info("update parameters: OK")


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    test_operand_factory()
    # test_update_parameters()
    # sys.exit()

    operand = op.operand_factory(op_defs.get('sma').get('def'))
    operand = op.operand_factory(op_defs.get('sma_of_rsi').get('def'))
    operand = op.operand_factory(
        ("ema", ("trix", {"timeperiod": 7},), {"timeperiod": 9})
    )

    operand.update_parameters(
        {
            'rsi_overbought_80.5': {
                'value': 90,
                'this_is_wrong': 'wrongest',
                'parameter_space': {'value': [80, 100]}
            },
        }
    )
    operand.update_parameters({'sma_10': {'timeperiod': 35}})
    logger.debug('before: %s', operand)
    operand.update_parameters({operand.unique_name: {'timeperiod': 129}})
    logger.debug('after: %s', operand)

    operand.run(data)

    print('-~•~-' * 40)
    print("indicator:")
    pprint(operand.indicator.plot_desc)
    print('-~•~-' * 40)
    print("operand:")
    pprint(operand.as_dict())
    # print(operand.indicator.unique_output)
    print('-~•~-' * 40)
    print("plot description:")
    pprint(operand.plot_desc)

    # sys.exit()

    logger.setLevel(logging.ERROR)
    runs = 1_000
    data = data
    st = time.time()

    opdef = op_defs.get('sma').get('def')

    for i in range(runs):
        operand = op.operand_factory(opdef)

    #     # operand.update_parameters({operand.unique_name: {'timeperiod': i}})
    #     # test_operand_run(op, data, False)

    # with Profile(timeunit=0.001) as p:
    #     for i in range(runs):
    #         _ = op.operand_factory(opdef)
    #         # op.update_parameters({op.unique_name: {'timeperiod': i}})

    # (
    #     Stats(p)
    #     .strip_dirs()
    #     .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
    #     # .reverse_order()
    #     .print_stats(30)
    # )

    # for _ in range(runs):
    #     test_execute_condition(data)

    print(f'length data: {len(data["close"])} periods')
    print(
        f"execution time: {(((time.time() - st) * 1_000_000) / runs):.2f} microseconds"
    )

    # pprint(operand.as_dict())
