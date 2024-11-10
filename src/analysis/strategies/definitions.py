#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:39:23 2023

@author_ dhaneor
"""
from random import randint, choice
from .. import strategy_builder as sb  # noqa: E402, F401
from . import signal_generator as sg  # noqa: E402, F401
from . import condition as cn  # noqa: E402, F401
from . import exit_order_strategies as es  # noqa: E402, F401

timeperiod = randint(50, 60)
trigger = 200

# Commoditiy Channel Index simple
cci = sg.SignalsDefinition(
    name=f"CCI {timeperiod}",
    conditions=cn.ConditionDefinition(
        interval="1d",
        operand_a=("cci", {"timeperiod": 70}),
        operand_b=("cci_oversold", -200, [-200, -70, 15]),
        operand_c=("cci_overbought", 67, [70, 200, 15]),
        open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
        open_short=("a", cn.COMPARISON.CROSSED_BELOW, "c"),
    ),
)

timeperiod = 10

rsi = sg.SignalsDefinition(
    name=f"RSI {timeperiod}",
    conditions=cn.ConditionDefinition(
        interval="1d",
        operand_a=("rsi", {"timeperiod": 2}),
        operand_b=("rsi_oversold", 182, [5, 35, 2]),
        operand_c=("rsi_overbought", 127, [65, 95, 2]),
        open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
        open_short=("a", cn.COMPARISON.CROSSED_BELOW, "c"),
    ),
)

timeperiod = 16

kama_cross = sg.SignalsDefinition(
    name=f"KAMA cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("kama", {"timeperiod": 2}),
            operand_b=("kama", {"timeperiod": 157}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

# BNB with REBALANCING
# best: params: (42, 0.22, 56, 62) :: risk level 7 :: max leverage 1,
# stats {'profit': 3152.497, 'max_drawdown': -32.189, 'sharpe_ratio': 1.691,
# 'sortino_ratio': 1.426, 'kalmar_ratio': 2.445, 'annualized_volatility': 38.814}

# BTC
# best: (21, 0.05, 66, 7) :: risk level 5 :: max leverage 1,
# stats {'profit': 1536.652, 'max_drawdown': -20.522, 'sharpe_ratio': 1.652,
# 'sortino_ratio': 1.146, 'kalmar_ratio': 2.893, 'annualized_volatility': 31.151}
#
# (70, 0.15, 12, 17) ... risk level 3 :: max leverage 1, (last 3 years from top)
#
# with leverage & REBALANCING: (21, 0.05, 67, 7) :: risk level 7(8) :: max leverage 1.5,
# stats {'profit': 2173.039, 'max_drawdown': -19.811, 'sharpe_ratio': 1.803,
# 'sortino_ratio': 1.28, 'kalmar_ratio': 3.45, 'annualized_volatility': 31.663}

# ETHUSDT
# best: params: params: (36, 0.07, 24, 7) :: risk level 6 :: max leverage 1,
# stats {'profit': 1928.265, 'max_drawdown': -31.241, 'sharpe_ratio': 1.338,
# 'sortino_ratio': 1.009, 'kalmar_ratio': 2.086, 'annualized_volatility': 45.148}
#
# with leverage & REBALANCING: (84, 0.05, 29, 12) :: risk level 7 :: max leverage 1.5
# {'profit': 3282.00105528086, 'max_drawdown': -33.46641732110709,
# 'sharpe_ratio': 1.7523866305577998, 'sortino_ratio': 1.6187130942382568,
# 'kalmar_ratio': 2.3868188824737215, 'annualized_volatility': 37.54528536133036}
breakout = sg.SignalsDefinition(
    name=f"Breakout {timeperiod}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 21}),
            operand_b=("trending", 0.05, [0.05, 0.55, 0.1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
            # close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
        ),
        cn.ConditionDefinition(
            interval="1d",
            operand_a="close",
            operand_b=("max", {"timeperiod": 66}),
            operand_c=("min", {"timeperiod": 7}),
            open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
            close_long=("a", cn.COMPARISON.IS_EQUAL, "c"),
            # open_short=("a", cn.COMPARISON.IS_EQUAL, "d"),
            # close_short=("a", cn.COMPARISON.IS_EQUAL, "b"),
        ),
    ]
)

timeperiod = 10
linreg_timeperiod = 20

trix = sg.SignalsDefinition(
    name=f"TRIX {timeperiod}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "linearreg_slope",
                ("trix", {"timeperiod": timeperiod}),
                {"timeperiod": linreg_timeperiod},
            ),
            operand_b=(
                "linearreg_slope",
                ("trix", {"timeperiod": timeperiod * 4}),
                {"timeperiod": linreg_timeperiod},
            ),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ],
)


linreg_roc_btc_1d = sg.SignalsDefinition(
    name="ROC - Linear Regression",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "roc",
                ("linearreg", "close", {"timeperiod": 42}),
                {"timeperiod": 27},
            ),
            operand_b=(
                "roc",
                ("linearreg", "close", {"timeperiod": 130}),
                {"timeperiod": 190},
            ),
            operand_c=("value", -6, [-21, 21, 1]),
            operand_d=("value", -21, [-21, 21, 1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)

linreg_roc_eth_1d = sg.SignalsDefinition(
    name="ROC 57 - Linear Regression 7",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "roc",
                ("linearreg", "close", {"timeperiod": 37}),
                {"timeperiod": 22},
            ),
            operand_b=(
                "roc",
                ("linearreg", "close", {"timeperiod": 42}),
                {"timeperiod": 27},
            ),
            operand_c=("value", -7, [-21, 21, 1]),
            operand_d=("value", -11, [-21, 21, 1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)

linreg = sg.SignalsDefinition(
    name=f"Linear Regression Slope {linreg_timeperiod}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "linearreg_slope",
                "close",
                {"timeperiod": 22},
            ),
            operand_b=(
                "linearreg_slope",
                "close",
                {"timeperiod": 7},
            ),
            operand_c=("slope", 0, [-10, 10, 2]),
            operand_d=("slope", 0, [-10, 10, 2]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)


timeperiod = randint(30, 50)
timeperiod = 8

ema_cross = sg.SignalsDefinition(
    name=f"EMA cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 7}),
            operand_b=("ema", {"timeperiod": 82}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

timeperiod = 30

tema_cross = sg.SignalsDefinition(
    name=f"TEMA cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("tema", {"timeperiod": 7}),
            operand_b=("tema", {"timeperiod": 82}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

# parameters for noise filtered KAMA
# 1) long-short
btc = (128, 0.05, 103), 4, 1
eth = (91, 0.15, 31), 6, 1
ada = (12, 0.1, 37), 3, 1
# 2) long only
btc = (7, 0.42, 117), 5, 1
eth = (7, 0.45, 37), 6, 1

test_er = sg.SignalsDefinition(
    name="KAMA with Noise Filter",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 7}),
            operand_b=("trending", 0.42, [0.05, 0.55, 0.1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
            # close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
        ),
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("close"),
            operand_b=("kama", {"timeperiod": 117}),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_BELOW, "b"),
            # close_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
        ),
    ]
)

# ======================================================================================
#                                       STRATEGIES                                     #
# ======================================================================================
contra_1 = sb.StrategyDefinition(
    strategy="Composite Strategy",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="RSI",
            symbol="BTCUSDT",
            interval="1d",
            signals_definition=rsi,
            weight=0.3,
        ),
        # sb.StrategyDefinition(
        #     strategy="RSI",
        #     symbol="BTCUSDT",
        #     interval="1d",
        #     signals_definition=rsi,
        #     weight=0.3,
        # ),
    ]
)

trend_1 = sb.StrategyDefinition(
    strategy="EMA Cross",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="EMA CROSS",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=ema_cross,
            weight=1,
        ),
    ]
)

s_tema_cross = sb.StrategyDefinition(
    strategy="TEMA Cross",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="TEMA CROSS",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=tema_cross,
            weight=1,
        ),
    ]
)

s_kama_cross = sb.StrategyDefinition(
    strategy="KAMA Cross",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="TEMA CROSS",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=kama_cross,
            weight=1,
        ),
    ]
)

s_breakout = sb.StrategyDefinition(
    strategy="Breakout",
    symbol="BTC/USDT",
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Breakout",
            symbol="BTC/USDT",
            interval="1d",
            signals_definition=breakout,
            weight=1,
            # stop_loss=(
            #     es.StopLossDefinition(
            #         strategy='atr',
            #         params=dict(atr_lookback=21, atr_factor=6)
            #     ),
            # )
        ),
    ],
)

s_trix = sb.StrategyDefinition(
    strategy="TRIX",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="trix",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=trix,
            weight=1,
        ),
    ]
)

s_linreg = sb.StrategyDefinition(
    strategy="Linear Regression (ROC)",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Linear Regression",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=linreg,
            weight=1,
        ),
    ]
)

s_test_er = sb.StrategyDefinition(
    strategy="KAMA with Noise Filter",
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Test ER",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=test_er,
            weight=1,
        ),
    ]
)


def get_all_strategies():
    return [contra_1, trend_1]
