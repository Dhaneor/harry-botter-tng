#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 23:05:20 2023

@author dhaneor
"""
from analysis import strategy_builder as sb
from analysis.strategy import operand as op
from analysis.strategy import condition as cnd
from analysis.strategy import signal_generator as sg
from analysis.strategy import exit_order_strategies as es
from analysis.strategy.definitions import cci, ema_cross  # noqa: F401


def test_build_single_strategy():
    sdef = sb.StrategyDefinition(
        strategy='test_strategy',
        symbol='BTCUSDT',
        interval='1m',
        signals_definition=sg.CrossSignal(
            short_signal=sg.CrossAboveSignal(fast_period=10, slow_period=30),
            long_signal=sg.CrossBelowSignal(fast_period=10, slow_period=30),
        ),
        weight=1,
        params={'param_1': 'value_1'},
        sub_strategies=None,
        stop_loss=[
            es.StopLossAtPrice(price=10000, order_type='MARKET'),
            es.StopLossAtPercentage(percentage=-0.05, order_type='MARKET'),
        ],
        take_profit=[
            es.TakeProfitAtPrice(price=10500, order_type='MARKET'),
            es.TakeProfitAtPercentage(percentage=0.05, order_type='MARKET'),
        ],
    )

    strategy = sb.build_single_strategy(sdef)

    assert strategy.NAME == 'test_strategy'
    assert strategy.symbol == 'BTCUSDT'
    assert strategy.interval == '1m'
    assert strategy.weight == 1
    assert strategy.params == {'param_1': 'value_1'}
    assert strategy.sl_strategy == [
        es.StopLossAtPrice(price=10000, order_type='MARKET'),
        es.StopLossAtPercentage(percentage=-0.05, order_type='MARKET'),
    ]
    assert strategy.tp_strategy == [
        es.TakeProfitAtPrice(price=10500, order_type='MARKET'),
        es.TakeProfitAtPercentage(percentage=0.05, order_type='MARKET'),
    ]

    assert isinstance(strategy._signal_generator, sg.CrossSignal)
    assert strategy._signal_generator.short_signal == sg.CrossAboveSignal(
        fast_period=10, slow_period=30
    )
    assert strategy._signal_generator.long_signal == sg.CrossBelowSignal(
        fast_period=10, slow_period=30
    )


def test_build_strategy():
    sdef = sb.StrategyDefinition(
        strategy='test_strategy',
        symbol='BTCUSDT',
        interval='1m',
        signals_definition=sg.CrossSignal(
            short_signal=sg.CrossAboveSignal(fast_period=10, slow_period=30),
            long_signal=sg.CrossBelowSignal(fast_period=10, slow_period=30),
        ),
        weight=1,
        params={'param_1': 'value_1'},
        sub_strategies=[
            sb.StrategyDefinition(
                strategy='test_sub_strategy_1',
                symbol='BTCUSDT',
                interval='1m',
                signals_definition=sg.CrossSignal(
                    short_signal=sg.CrossAboveSignal(fast_period=5, slow_period=10),
                    long_signal=sg.CrossBelowSignal(fast_period=5, slow_period=10),
                ),
                weight=0.5,
                params={'param_2': 'value_2'},
                sub_strategies=None,
                stop_loss=None,
                take_profit=None,
            ),
            sb.StrategyDefinition(
                strategy='test_sub_strategy_2',
                symbol='BTCUSDT',
                interval='1m',
                signals_definition=sg.CrossSignal(
                    short_signal=sg.CrossAboveSignal(fast_period=15, slow_period=30),
                    long_signal=sg.CrossBelowSignal(fast_period=15, slow_period=30),
                ),
                weight=0.5,
                params={'param_3': 'value_3'},
                sub_strategies=None,
                stop_loss=None,
                take_profit=None,
            ),
        ],
        stop_loss=[
            es.StopLossAtPrice(price=10000, order_type='MARKET'),
            es.StopLossAtPercentage(percentage=-0.05, order_type='MARKET'),
        ],
        take_profit=[
            es.TakeProfitAtPrice(price=10500, order_type='MARKET'),
            es.TakeProfitAtPercentage(percentage=0.05, order_type='MARKET'),
        ],
    )
