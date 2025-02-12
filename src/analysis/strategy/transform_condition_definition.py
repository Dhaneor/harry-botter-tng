#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a decorator for transforming Signalsfinition to 
SignalGeneratorDefinition instances.

SignalsDefinition = the previously used way to define signals.
SignalGeneratorDefinition = the 'new' new way to define signals.

This is just for convenience and to prevent having to rewrite
existing strategies all at once.

NOTE:
This module was just used to develop the decorator. The code has been
copied to the signal_generator.py module and is available there, no
need for imports.

Created on Jan 03 10:03:20 2025

@author dhaneor
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pprint import pprint
from typing import Callable, Any

from analysis.strategy.condition import ConditionDefinition, COMPARISON
from util import DotDict


@dataclass
class SignalsDefinition:
    """Definition for how to produce entry/exit signals.

    Attributes:
    -----------
    name: str
        the name of the signal definition
    open_long: ConditionDefinitionsT
        A sequence of conditions that must all be True to open a long position.
        For convenience, intead of a sequence, a single ConditionDefinition
        can be passed. So, you can pass here:
        - Sequence[cnd.ConditionDefinition]
        - cnd.ConditionDefinition
        - dict[PositionTypeT, cnd.ConditionDefinition]

    open_short: ConditionDefinitionsT
        a sequence of conditions that must all be True to open a short position

    close_long: ConditionDefinitionsT
        a sequence of conditions that must all be True to close a long position

    close_short: ConditionDefinitionsT
        a sequence of conditions that must all be True to close a short position

    reverse: bool
        just use reversed long condition for shorts, default: False
    """
    name: str = "unnamed"
    conditions: list = None

    def __repr__(self):
        out = [f"SignalsDefinition: {self.name}\n"]
        [out.append(f"\t{c}\n") for c in self.conditions if c is not None]

        return "".join(out)


aroon_osc = SignalsDefinition(
    name="AROON OSC",
    conditions=[
        ConditionDefinition(
            interval="1d",
            operand_a=("aroonosc", "high", "low", {"timeperiod": 4}),
            operand_b=("aroon_trigger", 1, [-5, 5, 1]),
            open_long=("a", COMPARISON.CROSSED_ABOVE, "b"),
            close_long=("a", COMPARISON.CROSSED_BELOW, "b"),
        ),
        ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 37}),
            operand_b=("er_trending", 0.21, [0.05, 0.55, 0.1]),
            open_long=("a", COMPARISON.IS_ABOVE, "b"),
            close_long=("a", COMPARISON.IS_BELOW, "b"),
        ),
    ],
)


ema_cross = SignalsDefinition(
    name="EMA cross",
    conditions=[
        ConditionDefinition(
            interval="1d",
            operand_a=("ema", {"timeperiod": 47}),
            operand_b=("ema", {"timeperiod": 182}),
            open_long=("a", COMPARISON.CROSSED_ABOVE, "b"),
            open_short=("a", COMPARISON.CROSSED_BELOW, "b"),
        ),
    ]
)


test_er = SignalsDefinition(
    name="KAMA with Noise Filter",
    conditions=[
        ConditionDefinition(
            interval="1d",
            operand_a=("er", {"timeperiod": 7}),
            operand_b=("trending", 0.01, [0.005, 0.055, 0.005]),
            open_long=("a", COMPARISON.IS_ABOVE, "b"),
            close_long=("a", COMPARISON.IS_BELOW, "b"),
            open_short=("a", COMPARISON.IS_ABOVE, "b"),
            close_short=("a", COMPARISON.IS_BELOW, "b"),
        ),
        ConditionDefinition(
            interval="1d",
            operand_a=("close"),
            operand_b=("kama", {"timeperiod": 9}),
            open_long=("a", COMPARISON.IS_ABOVE, "b"),
            close_long=("a", COMPARISON.IS_BELOW, "b"),
            open_short=("a", COMPARISON.IS_BELOW, "b"),
            close_short=("a", COMPARISON.IS_ABOVE, "b"),
        ),
    ]
)


class SignalGeneratorDefinition(DotDict):
    ...


aroon_osc_new = SignalGeneratorDefinition(
    name="AROON OSC (New)",
    operands=dict(
        aroonosc=("aroonosc", "high", "low", {"timeperiod": 4}),
        aroon_trigger=("aroon_trigger", 1, [-5, 5, 1]),
        er=("er", {"timeperiod": 37}),
        er_trending=("er_trending", 0.21, [0.05, 0.55, 0.1]),
    ),
    conditions=dict(
        open_long=[
            ("aroon", COMPARISON.CROSSED_ABOVE, "aroon_trigger"),
            ("efficiency_ratio", COMPARISON.IS_ABOVE, "er_trending"),
        ],
        close_long=[
            ("aroon", COMPARISON.CROSSED_BELOW, "aroon_trigger"),
            ("efficiency_ratio", COMPARISON.IS_BELOW, "er_trending"),
        ],
        open_short=None,
        close_short=None,
    ),
)


def transform_condition_definition(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], DotDict):
            return func(*args, **kwargs)

        if args:  #  and isinstance(args[0], SignalsDefinition):
            signals_def = args[0]

            try:
                if 'operands' in signals_def.operands:
                    return func(*args, **kwargs)
            except:
                pass

            operands = {}
            conditions = {
                "open_long": [],
                "close_long": [],
                "open_short": [],
                "close_short": [],
            }
            indicator_counters = defaultdict(int)
            indicator_names = {}

            def get_unique_operand_name(operand_value):
                if isinstance(operand_value, tuple):
                    indicator_name = operand_value[0]
                    if indicator_name in ['open', 'high', 'low', 'close', 'volume']:
                        return indicator_name

                    # Check if we've already assigned a unique name to this exact indicator
                    indicator_key = str(operand_value)
                    if indicator_key in indicator_names:
                        return indicator_names[indicator_key]

                    indicator_counters[indicator_name] += 1
                    unique_name = f"{indicator_name}_{indicator_counters[indicator_name]}"
                    indicator_names[indicator_key] = unique_name
                    return unique_name
                return operand_value

            for condition in signals_def.conditions:
                for operand in ["a", "b", "c", "d"]:
                    operand_attr = f"operand_{operand}"
                    if hasattr(condition, operand_attr):
                        operand_value = getattr(condition, operand_attr)
                        if operand_value:
                            unique_name = get_unique_operand_name(operand_value)
                            operands[unique_name] = operand_value

                for condition_type in ["open_long", "close_long", "open_short", "close_short"]:
                    if hasattr(condition, condition_type):
                        condition_value = getattr(condition, condition_type)
                        if condition_value:
                            transformed_condition = list(condition_value)
                            for i, item in enumerate(transformed_condition):
                                if item in ["a", "b", "c", "d"]:
                                    operand_attr = f"operand_{item}"
                                    if hasattr(condition, operand_attr):
                                        operand_value = getattr(condition, operand_attr)
                                        unique_name = get_unique_operand_name(operand_value)
                                        transformed_condition[i] = unique_name
                            conditions[condition_type].append(tuple(transformed_condition))

            transformed_def = SignalGeneratorDefinition(
                name=signals_def.name,
                operands=operands,
                conditions={k: v if v else None for k, v in conditions.items()},
            )
            args = (transformed_def,) + args[1:]

        else:
            print(f"Invalid input: {func.__name__} expects a SignalsDefinition")
            print(f"Got: {args} (type: {type(args[0])})")

        return func(*args, **kwargs)

    return wrapper

@transform_condition_definition
def dummy_test_function(signals_def: SignalsDefinition) -> None:
    print(f"Testing {signals_def.name}")
    pprint(signals_def)
    print("-" * 80)

    signals_def.conditions.test = ("test", "a", "b", "c")
    print("Updated conditions: ", signals_def.conditions.test)

    for k,v in signals_def.conditions.items():
        print(f"{k}:")
        for condition in v:
            print(f"  - {condition}")
        print()


if __name__ == "__main__":
    dummy_test_function(test_er)
