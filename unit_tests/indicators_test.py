#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import sys
import os
import time
import pandas as pd
from pprint import pprint
import numpy as np
from cProfile import Profile  # noqa: E402, F401
import pstats  # noqa: E402, F401
import logging

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import src.analysis.indicators.indicator as indicator  # noqa: E402

# configure logger
LOG_LEVEL = logging.DEBUG
logger = logging.getLogger('main')
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s'
)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.info('Logger initialized')


# import sample data for BTCUSDT 15min
file_name = os.path.join(parent, 'ohlcv_data', 'btcusdt_15m.csv')
data = pd.read_csv(file_name, index_col=0, parse_dates=True)

# clean the ohlcv data
data.drop(['human open time', 'quote asset volume', 'close time'], axis=1, inplace=True)
data['open time'] = pd.to_datetime(data['open time'], unit='ms')
data.set_index(keys=['open time'], inplace=True)

# data = data.resample('1d')\
#     .agg(
#             {
#             'close': 'last', 'open': 'first',
#             'high': 'max', 'low': 'min', 'volume': 'sum',
#             },
#             min_periods=1
#     )

data = data[-1000:]
arr1 = np.array([data.close.tolist() for _ in range(50)]).T
arr = data.close.to_numpy()

array_len = 1000

a = np.random.rand(array_len) * 100
a1 = np.random.rand(array_len) * 1.05
a2 = np.random.rand(array_len) * 0.95

# ==============================================================================
defs = {
    'SMA': {
        'src': 'talib',
        'params': {'timeperiod': 100},
        "plot_desc": indicator.PlotDescription(
            label='sma_100',
            is_subplot=False,
            lines=[('sma_100', 'Line')],
            triggers=[],
            channel=[],
            hist=[],
            level='indicator'
        )
    },
    "STOCH": {
        "src": "talib",
        "params": {
            "fastk_period": 14,
            "slowk_period": 5,
            "slowd_period": 5,
            "slowk_matype": 0,
            "slowd_matype": 0,
        },
        "plot_desc": indicator.PlotDescription(
            label='stoch_14_5_0_5_0',
            is_subplot=True,
            lines=[
                ('stoch_14_5_0_5_0_slowk', 'Dashed Line'),
                ('stoch_14_5_0_5_0_slowd', 'Dashed Line')
            ],
            triggers=[],
            channel=[],
            hist=[],
            level='indicator'
        )
    },
    "BBANDS": {
        "src": "talib",
        "params": {
            "timeperiod": 20,
            "nbdevup": 2,
            "nbdevdn": 2,
            "matype": 0,
        },
        "plot_desc": indicator.PlotDescription(
            label='bbands_20_2.0_2.0_0',
            is_subplot=False,
            lines=[('bbands_20_2.0_2.0_0_middleband', 'Line')],
            triggers=[],
            channel=['bbands_20_2.0_2.0_0_upperband', 'bbands_20_2.0_2.0_0_lowerband'],
            hist=[],
            level='indicator'
        )
    },
    "MACD": {
        "src": "talib",
        "params": {
            "fastperiod": 12,
            "slowperiod": 26,
            "signalperiod": 9
        },
        "plot_desc": indicator.PlotDescription(
            label='macd_12_26_9',
            is_subplot=True,
            lines=[
                ('macd_12_26_9_macd', 'Line'),
                ('macd_12_26_9_macdsignal', 'Dashed Line')
            ],
            triggers=[],
            channel=[],
            hist=['macd_12_26_9_macdhist'],
            level='indicator'
        )
    },
    "RSI_OVERBOUGHT": {
        "src": "fixed",
        "params": {'value': 70},
        'parameter_space': {'trigger': [70, 100]},
        "plot_desc": indicator.PlotDescription(
            label='rsi_overbought_70',
            is_subplot=True,
            lines=[],
            triggers=[('rsi_overbought_70', 'Line')],
            channel=[],
            hist=[],
            level='indicator'
        )
    },
    "RSI_OVERSOLD": {
        "src": "fixed",
        "params": {'value': 30},
        'parameter_space': {'trigger': [0, 30]},
        "plot_desc": indicator.PlotDescription(
            label='rsi_oversold_30',
            is_subplot=True,
            lines=[],
            triggers=[('rsi_oversold_30', 'Line')],
            channel=[],
            hist=[],
            level='indicator'
        )
    },
}


# ==============================================================================
def test_indicator_factory(name: str, params: dict | None = None,
                           src: str = 'talib', show: bool = False
                           ) -> indicator.Indicator:

    ind = indicator.factory(name, params, src)

    if show:
        print('name:', ind.name)
        print('input:', ind.input)
        print('params:', ind.parameters)
        print('out:', ind.output)
        print(ind)
        ind.help()

    assert isinstance(ind, indicator.IIndicator)
    assert ind.input is not None
    assert ind.parameters is not None
    assert ind.output is not None
    return ind


def test_set_indicator_parameters():
    for cand in defs:
        i = indicator.factory(
            indicator_name=cand, params=defs[cand]['params'], source=defs[cand]['src']
        )

        if params := defs[cand].get('params'):
            for k, v in params.items():
                logger.info("setting parameter %s for %s -> %s", k, i, v * 2)
                i.parameters = {k: v * 2}
                assert i.parameters[k] == v * 2

        logger.debug(i)


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
            raise ValueError(f'invalid input: {i.input}')

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


def test_plot_desc():
    logger.setLevel(logging.INFO)

    for cand in defs:
        i = indicator.factory(cand, defs[cand]['params'], defs[cand]['src'])
        logger.info(i.plot_desc)

        try:
            assert i.plot_desc == defs[cand]['plot_desc']
        except AssertionError as e:
            logger.error(f'plot description mismatch for {cand} --> %s', e)
            logger.error("expected:\t%s", defs[cand]['plot_desc'])
            logger.error("got:     \t%s", i.plot_desc)
            return

    logger.info("verify plot descriptions: OK")


# tests for the Parameter class
def test_parameter():
    """Tests the Parameter class"""
    p = indicator.Parameter(
        name='slowperiod',
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
                assert p.value == int(round(v)), \
                    f"expected {int(round(v))}, but got {p.value}"
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
        name='slowperiod',
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
                p.min_, p.max_, min_, max_
            )
        except ValueError as e:
            logger.error(
                "set parameter space to [%s, %s]: FAIL -> %s", min_, max_, e
            )

    logger.info("finished: OK")


def test_parameter_iter():
    p = indicator.Parameter(
        name='test',
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


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':
    # test_set_indicator_parameters()

    # test_parameter_space()
    # test_parameter_iter()
    # test_plot_desc()
    # sys.exit()

    # arr, res = test_is_above()

    # ind = test_indicator_factory(
    #     'BBANDS', {'timeperiod': 80, 'nbdevup': 1.5},
    #     show=False
    # )

    ind = test_indicator_factory(
        'LINEARREG', params={'timeperiod': 20}, show=True
        )

    # ind = test_indicator_factory('SMA', {'timeperiod': 20}, show=False)

    # ind = test_indicator_factory('STOCH', show=False)

    # ind = test_indicator_factory(
    #     'rsi_oversold',
    #     {'value': 70, 'parameter_space': {'trigger': [30, 70]}},
    #     'fixed',
    #     show=False
    # )

    # ind.parameters = {'value': 80, 'parameter_space': [40, 70]}

    print(ind.help())

    # print(ind)
    # pprint(ind.__dict__)
    # print(ind.unique_output)
    print(ind.plot_desc)

    # print(test_indicator_run(ind))

    sys.exit(0)

    runs = 1_000_000
    logger.setLevel(logging.ERROR)
    st = time.time()

    # for _ in range(runs):
    #     ind = test_indicator_factory(
    #         'BBANDS', {'timeperiod': 80, 'nbdevup': 1.5},
    #         show=False
    #     )
    #     ind.parameter_space = {'value': [40, 70]}

    with Profile() as p:
        for _ in range(runs):
            ind.parameter_space = {'value': [40, 70]}

        (
            pstats.Stats(p)
            .strip_dirs()
            .sort_stats(pstats.SortKey.CUMULATIVE)
            # .reverse_order()
            .print_stats(20)
        )

    print(
        f'execution time: {((time.time() - st)*1_000_000/runs):.2f} microseconds'
    )
