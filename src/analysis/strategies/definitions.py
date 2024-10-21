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

timeperiod = randint(50, 60)
trigger = 200

# Commoditiy Channel Index simple
cci = sg.SignalsDefinition(
    name=f"CCI {timeperiod}",
    conditions=cn.ConditionDefinition(
        interval="1d",
        operand_a=(
            "cci",
            {"timeperiod": timeperiod},
        ),
        operand_b=(
            "cci_oversold",
            trigger * -1,
            {"parameter_space": {"trigger": [70, 200, 1]}},
        ),
        operand_c=(
            "cci_overbought",
            trigger,
            {"parameter_space": {"trigger": [70, 200, 1]}},
        ),
        open_long=("a", cn.COMPARISON.IS_BELOW, "b"),
        open_short=("a", cn.COMPARISON.IS_ABOVE, "c"),
    ),
)

timeperiod = 10

rsi = sg.SignalsDefinition(
    name=f"RSI {timeperiod}",
    conditions=cn.ConditionDefinition(
        interval="1d",
        operand_a=("rsi", {"timeperiod": timeperiod}),
        operand_b=("rsi_oversold", 30, {"parameter_space": {"trigger": [5, 35]}}),
        operand_c=("rsi_overbought", 70, {"parameter_space": {"trigger": [65, 95]}}),
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
            operand_a=("kama", {"timeperiod": timeperiod}),
            operand_b=("kama", {"timeperiod": timeperiod * 4}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

timeperiod = 20

breakout = sg.SignalsDefinition(
    name=f"Breakout {timeperiod}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a="close",
            operand_b=("max", {"timeperiod": timeperiod}),
            operand_c=("min", {"timeperiod": timeperiod}),
            open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
            open_short=("a", cn.COMPARISON.IS_EQUAL, "c"),
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

linreg_timeperiod = 15

linreg = sg.SignalsDefinition(
    name=f"TRIX {timeperiod}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "linearreg_slope",
                "close",  # ("trix", {"timeperiod": 3}),
                {"timeperiod": linreg_timeperiod},
            ),
            operand_b=(
                "slope",
                0,
                {"parameter_space": {"trigger": [-10, 10, 1]}},
                ),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ],
)


timeperiod = randint(30, 50)
timeperiod = 12

ema_cross = sg.SignalsDefinition(
    name=f"EMA cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": timeperiod}),
            operand_b=("ema", {"timeperiod": timeperiod * 6}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

timeperiod = 20

tema_cross = sg.SignalsDefinition(
    name=f"TEMA cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("tema", {"timeperiod": timeperiod}),
            operand_b=("tema", {"timeperiod": timeperiod * 4}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
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
            strategy="CCI",
            symbol="BTCUSDT",
            interval="1d",
            signals_definition=cci,
            weight=0.3,
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
    symbol=choice(("BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT")),
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Breakout",
            symbol="ETHUSDT",
            interval="1d",
            signals_definition=breakout,
            weight=1,
        ),
    ]
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
    strategy="Linear Regression",
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


def get_all_strategies():
    return [contra_1, trend_1]
