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
cci = sg.SignalGeneratorDefinition(
    name="CCI simple",
    operands=dict(
        cci=("cci")  ,# , {"timeperiod": 21}),
        oversold=("oversold", -150, [-200, -70, 15]),
        overbought=("overbought", 150, [70, 200, 15]),
        rsi=("rsi", {"timeperiod": 14}),
    ),
    conditions=dict(
        open_long=[
            ("cci", cn.COMPARISON.IS_BELOW, "oversold")
        ],
        close_long=[
            ("cci", cn.COMPARISON.IS_ABOVE, "overbought")
        ]
    ),
)


rsi = sg.SignalsDefinition(
    name="RSI simple",
    conditions=cn.ConditionDefinition(
        interval="1d",
        operand_a=("rsi", {"timeperiod": 2}),
        operand_b=("rsi_oversold", 20, [5, 35, 2]),
        operand_c=("rsi_overbought", 80, [65, 95, 2]),
        open_long=("a", cn.COMPARISON.IS_BELOW, "b"),
        close_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
    ),
)


kama_cross = sg.SignalsDefinition(
    name="KAMA cross",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("kama", {"timeperiod": 87}),
            operand_b=("kama", {"timeperiod": 97}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
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
#
# TIKR MVP: params: (90, 0.12, 7, 3) :: risk level 9 :: max leverage 1,
# {'profit': 3838.216, 'max_drawdown': -25.471, 'sharpe_ratio': 1.80,
# 'sortino_ratio': 1.39, 'kalmar_ratio': 3.31, 'annualized_volatility': 37.903}

# ETHUSDT
# best params: (36, 0.07, 24, 7) :: risk level 6 :: max leverage 1,
# stats {'profit': 1928.265, 'max_drawdown': -31.241, 'sharpe_ratio': 1.338,
# 'sortino_ratio': 1.009, 'kalmar_ratio': 2.086, 'annualized_volatility': 45.148}
#
# with leverage & REBALANCING: (84, 0.05, 29, 12) :: risk level 7 :: max leverage 1.5
# {'profit': 3282.00105528086, 'max_drawdown': -33.46641732110709,
# 'sharpe_ratio': 1.7523866305577998, 'sortino_ratio': 1.6187130942382568,
# 'kalmar_ratio': 2.3868188824737215, 'annualized_volatility': 37.54528536133036}
# breakout = sg.SignalsDefinition(
#     name=f"Breakout {timeperiod}",
#     conditions=[
#         # cn.ConditionDefinition(
#         #     interval="1d",
#         #     operand_a=("er", {"timeperiod": 20}),
#         #     operand_b=("trending", 0.2, [0.05, 0.55, 0.1]),
#         #     open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
#         #     close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
#         #     open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
#         #     close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
#         # ),
#         cn.ConditionDefinition(
#             interval="1d",
#             operand_a="close",
#             operand_b=("max", {"timeperiod": 20}),
#             operand_c=("min", {"timeperiod": 12}),
#             operand_d=("max", {"timeperiod": 7}),
#             open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
#             close_long=("a", cn.COMPARISON.IS_EQUAL, "c"),
#             open_short=("a", cn.COMPARISON.IS_EQUAL, "c"),
#             close_short=("a", cn.COMPARISON.IS_EQUAL, "d"),
#         ),
#     ]
# )

# best for ETHUSDT (test: 16-02-2025)
breakout = sg.SignalsDefinition(
    name="TIKR MVP Signal",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 10, "smoothing": 10}),
            operand_b=("trending", 0.05, [0.05, 0.55, 0.1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
        ),
        cn.ConditionDefinition(
            interval="1d",
            operand_a="close",
            operand_b=("max", {"timeperiod":57}),
            operand_c=("min", {"timeperiod": 2}),
            open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
            close_long=("a", cn.COMPARISON.IS_EQUAL, "c"),
        ),
    ]
)

# best for BTCUSDT (test: 13-02-2025)
# breakout = sg.SignalsDefinition(
#     name="TIKR MVP Signal",
#     conditions=[
#         cn.ConditionDefinition(
#             interval="1d",
#             operand_a=("er", {"timeperiod": 95, "smoothing": 7}),
#             operand_b=("trending", 0.15, [0.05, 0.55, 0.1]),
#             open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
#             close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
#         ),
#         cn.ConditionDefinition(
#             interval="1d",
#             operand_a="close",
#             operand_b=("max", {"timeperiod":2}),
#             operand_c=("min", {"timeperiod": 27}),
#             open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
#             close_long=("a", cn.COMPARISON.IS_EQUAL, "c"),
#         ),
#     ]
# )

# ... this is the one that was used initially for TKIR MVP
# breakout = sg.SignalsDefinition(
#     name="TIKR MVP Signal",
#     conditions=[
#         cn.ConditionDefinition(
#             interval="1d",
#             operand_a=("er", {"timeperiod": 90, "smoothing": 1}),
#             operand_b=("trending", 0.12, [0.05, 0.55, 0.1]),
#             open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
#             close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
#         ),
#         cn.ConditionDefinition(
#             interval="1d",
#             operand_a="close",
#             operand_b=("max", {"timeperiod": 7}),
#             operand_c=("min", {"timeperiod": 3}),
#             open_long=("a", cn.COMPARISON.IS_EQUAL, "b"),
#             close_long=("a", cn.COMPARISON.IS_EQUAL, "c"),
#         ),
#     ]
# )



trix = sg.SignalsDefinition(
    name="TRIX",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                'rocp',
                ("linearreg_slope",  "close", {"timeperiod": 10}),
                {"timeperiod": 3},
            ),
            operand_b=('trigger', 0, [-5, 5, 1]),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ],
)

# BTCUSDT
# params: (4, 1, 37, 0.21) :: risk level 9 :: max leverage 1,
# stats {'profit': 5436.227, 'max_drawdown': -23.921, 'sharpe_ratio': 1.976,
# 'sortino_ratio': 1.535, 'kalmar_ratio': 3.983, 'annualized_volatility': 37.399}
#
# params: (4, 1, 37, 0.21) :: risk level 0 :: max leverage 1,
# stats {'profit': 7643.912, 'max_drawdown': -32.924, 'sharpe_ratio': 1.842,
# 'sortino_ratio': 1.401, 'kalmar_ratio': 3.235, 'annualized_volatility': 44.794}
#
# ETHUSDT
# (17, 3, 60, 0.05) :: risk level 5 :: max leverage 1.5,
# stats {'profit': 4427.212, 'max_drawdown': -25.013, 'sharpe_ratio': 2.04,
# 'sortino_ratio': 2.044, 'kalmar_ratio': 3.552, 'annualized_volatility': 34.035}
#
# not worth it with risk level 0!
#
# SOLUSDT
# params: (3, 1, 15, 0.31) :: risk level 9 :: max leverage 1.5,
# stats {'profit': 2256.527, 'max_drawdown': -20.569, 'sharpe_ratio': 2.362,
# 'sortino_ratio': 1.82, 'kalmar_ratio': 6.305, 'annualized_volatility': 38.314}
#
# ADAUSDT
# params: (2, 5, 37, 0.33) :: risk level 0 :: max leverage 1,
# stats {'profit': 7427.879, 'max_drawdown': -35.107, 'sharpe_ratio': 1.671,
# 'sortino_ratio': 0.988, 'kalmar_ratio': 3.007, 'annualized_volatility': 50.368}
aroonosc = sg.SignalsDefinition(
    name="AROON OSC",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=('aroonosc', 'high', 'low', {"timeperiod":47}),
            operand_b=('trigger', -3, [-5, 5, 1]),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 65, "smoothing": 1}),
            operand_b=("trending", 0.15, [0.05, 0.55, 0.1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
            # close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
        ),
    ],
)

# BTC long only min drawdown: (37, 17, 18) :: risk level 8 :: max leverage 1.5,
# stats {'profit': 2354.18, 'max_drawdown': -21.785, 'sharpe_ratio': 1.836,
# 'sortino_ratio': 1.171, 'kalmar_ratio': 3.237, 'annualized_volatility': 31.824}

# BTC long-only (52, 2, 9) :: risk level 0 :: max leverage 1,
# stats {'profit': 3022.464, 'max_drawdown': -27.908, 'sharpe_ratio': 1.533,
# 'sortino_ratio': 1.124, 'kalmar_ratio': 2.777, 'annualized_volatility': 43.622}

# BTC long-only max profit: (56, 6, 4) :: risk level 0 :: max leverage 1,
# stats {'profit': 3901.009, 'max_drawdown': -34.671, 'sharpe_ratio': 1.571,
# 'sortino_ratio': 1.235, 'kalmar_ratio': 2.451, 'annualized_volatility': 45.833}

linreg_roc_btc_1d = sg.SignalsDefinition(
    name="ROC - Linear Regression",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "roc",
                ("linearreg", "close", {"timeperiod": 6}),
                {"timeperiod": 56},
            ),
            # operand_b=(
            #     "roc",
            #     ("linearreg", "close", {"timeperiod": 130}),
            #     {"timeperiod": 190},
            # ),
            operand_c=("value", 4, [-21, 21, 3]),
            # operand_d=("value", -21, [-21, 21, 1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            # open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            # close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)

linreg_roc_eth_1d = sg.SignalsDefinition(
    name="ROC 57 - Linear Regression 7",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            # operand_a=(
            #     "roc",
            #     ("linearreg", "close", {"timeperiod": 37}),
            #     {"timeperiod": 22},
            # ),
            operand_b=(
                "roc",
                ("linearreg", "close", {"timeperiod": 42}),
                {"timeperiod": 27},
            ),
            # operand_c=("value", -7, [-21, 21, 1]),
            operand_d=("value", -11, [-21, 21, 1]),
            # open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            # close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)


linreg_roc = sg.SignalsDefinition(
    name="ROCP of Linear Regression",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "rocp",
                ("linearreg", "close", {"timeperiod": 10}),
                {"timeperiod": 1},
            ),
            operand_b=(
                "rocp",
                ("linearreg", "close", {"timeperiod": 5}),
                {"timeperiod": 1},
            ),
            operand_c=(
                "rocp",
                ("linearreg", "close", {"timeperiod": 3}),
                {"timeperiod": 1},
            ),
            operand_d=("value", 0, [-21, 21, 1]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "d"),
            close_long=("b", cn.COMPARISON.IS_BELOW, "d"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("c", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)


linreg = sg.SignalsDefinition(
    name="Linear Regression Slope (long/short)",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "linearreg_slope",
                "close",
                {"timeperiod": 37},
            ),
            operand_b=(
                "linearreg_slope",
                "close",
                {"timeperiod": 7},
            ),
            operand_c=("trigger", 0, [-10, 10, 2]),
            operand_d=("trigger", 0, [-10, 10, 2]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "c"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "c"),
            open_short=("b", cn.COMPARISON.IS_BELOW, "d"),
            close_short=("b", cn.COMPARISON.IS_ABOVE, "d"),
        ),
    ],
)

linreg_ma_cross = sg.SignalsDefinition(
    name="Linear Regression Slope",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=(
                "linearreg",
                "close",
                {"timeperiod": 2},
            ),
            operand_b=(
                "linearreg",
                "close",
                {"timeperiod": 142},
            ),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ],
)


ema_cross = sg.SignalsDefinition(
    name="EMA Cross",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 47}),
            operand_b=("ema", {"timeperiod": 182}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)


tema_cross = sg.SignalsDefinition(
    name="TEMA Cross",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("tema", {"timeperiod": 18}),
            operand_b=("tema", {"timeperiod": 179}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

# parameters for noise filtered KAMA
# 1) long-short
btc = (128, 0.05, 103), 4, 1
eth = (14, 0.35, 2), 4, 1
ada = (12, 0.1, 37), 3, 1
# 2) long only
# BTC: (70, 0.2, 167) :: risk level 8 :: max leverage 1.25,
# stats {'profit': 1907.318, 'max_drawdown': -21.953, 'sharpe_ratio': 1.62,
# 'sortino_ratio': 1.088, 'kalmar_ratio': 2.956, 'annualized_volatility': 34.551}
#
# BTC: (49, 0.15, 182) :: risk level 8 :: max leverage 1,
# stats {'profit': 1905.738, 'max_drawdown': -29.57, 'sharpe_ratio': 1.6,
# 'sortino_ratio': 1.24, 'kalmar_ratio': 2.194, 'annualized_volatility': 35.088}
#
# BTC: (70, 0.2, 167) :: risk level 0 :: max leverage 1,
# stats {'profit': 2483.611, 'max_drawdown': -28.132, 'sharpe_ratio': 1.529,
# 'sortino_ratio': 1.017, 'kalmar_ratio': 2.559, 'annualized_volatility': 40.904}
#
# BTC: (49, 0.15, 182) :: risk level 0 :: max leverage 1,
# stats {'profit': 2766.9935, 'max_drawdown': -30.184, 'sharpe_ratio': 1.517,
# 'sortino_ratio': 1.1603, 'kalmar_ratio': 2.484, 'annualized_volatility': 42.923}
eth = (7, 0.45, 37), 7, 1  # profit: 1625.779; drawdown: -26.234
eth = (7, 0.45, 37), 9, 1.5  # profit': 3440.149, 'max_drawdown': -32.625

test_er = sg.SignalsDefinition(
    name="KAMA with Noise Filter",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 70}),
            operand_b=("trending", 0.2, [0.005, 0.055, 0.005]),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
            # close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
        ),
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("close"),
            operand_b=("kama", {"timeperiod": 167}),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_BELOW, "b"),
            # close_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
        ),
    ]
)


test_er_2 = sg.SignalsDefinition(
    name="Pure ER",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 20}),
            operand_b=("er", {"timeperiod": 40}),
            open_long=("a", cn.COMPARISON.IS_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.IS_BELOW, "b"),
            # open_short=("a", cn.COMPARISON.IS_ABOVE, "b"),
            # close_short=("a", cn.COMPARISON.IS_BELOW, "b"),
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

s_trend_1 = sb.StrategyDefinition(
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
    signals_definition=tema_cross,
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
            #         params=dict(atr_lookback=21, atr_factor=4)
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
            symbol="BTCUSDT",
            interval="1d",
            signals_definition=linreg_roc,
            weight=1,
        ),
    ]
)

s_test_er = sb.StrategyDefinition(
    strategy="KAMA with Noise Filter",
    symbol="BTC/USDT",
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Test ER",
            symbol="BTC/USDT",
            interval="1d",
            signals_definition=test_er,
            weight=1,
        ),
    ]
)

s_linreg_ma_cross = sb.StrategyDefinition(
    strategy="KAMA with Noise Filter",
    symbol="BTC/USDT",
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Test ER",
            symbol="BTC/USDT",
            interval="1d",
            signals_definition=linreg_ma_cross,
            weight=1,
        ),
    ]
)

s_aroon_osc = sb.StrategyDefinition(
    strategy="AROON oscillator with Noise Filter",
    symbol="BTCUSDT",
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="AROON oscillator",
            symbol="BTC/USDT",
            interval="1d",
            signals_definition=aroonosc,
            weight=1,
        ),
    ]
)

# ====================================================================================
#                                     MULTI STRATEGY 1                               #
# ====================================================================================
ema_cross_10_30 = sg.SignalsDefinition(
    name=f"ema cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 22}),
            operand_b=("ema", {"timeperiod": 67}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

ema_cross_20_60 = sg.SignalsDefinition(
    name=f"ema cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 2}),
            operand_b=("ema", {"timeperiod": 87}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

ema_cross_40_120 = sg.SignalsDefinition(
    name=f"ema cross {timeperiod}/{timeperiod*4}",
    conditions=[
        cn.ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 32}),
            operand_b=("ema", {"timeperiod": 52}),
            open_long=("a", cn.COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", cn.COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)

s_test_ema_cross = sb.StrategyDefinition(
    strategy="Test for optimizer",
    symbol="BTCUSDT",
    interval="1d",
    sub_strategies=[
        sb.StrategyDefinition(
            strategy="Optimizer Test",
            symbol="BTC/USDT",
            interval="1d",
            signals_definition=ema_cross_10_30,
            weight=1,
        ),
    ]
)

s_ema_multi = sb.StrategyDefinition(
        strategy="Composite Strategy",
        symbol="BTCUSDT",
        interval="1d",
        sub_strategies=[
            sb.StrategyDefinition(
                strategy="Sub 1",
                symbol="BTCUSDT",
                interval="1d",
                signals_definition=ema_cross_10_30,
                weight=0.33,
            ),
            sb.StrategyDefinition(
                strategy="Sub 2",
                symbol="BTCUSDT",
                interval="1d",
                signals_definition=ema_cross_20_60,
                weight=0.33,
            ),
            sb.StrategyDefinition(
                strategy="Sub 3",
                symbol="BTCUSDT",
                interval="1d",
                signals_definition=ema_cross_40_120,
                weight=0.33,
            ),
        ]
    )


# ====================================================================================
#                                   END OF DEFINitiONS                               #
# ====================================================================================
def get_all_strategies():
    return [contra_1, s_trend_1]
