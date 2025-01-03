#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""

import yaml
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any

from analysis.strategy.condition import ConditionDefinition, COMPARISON


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


class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        return self.to_yaml()

    def __str__(self):
        return self.to_yaml()

    def to_yaml(self):
        """Convert the DotDict to a YAML string, including nested values."""
        return yaml.dump(
            self.to_dict(), default_flow_style=False, sort_keys=False, indent=4
        )

    def to_dict(self):
        """Recursively convert DotDict (and nested DotDicts) to standard dictionaries"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):  # Check if the value is a DotDict
                result[key] = value.to_dict()  # Recursively convert nested DotDict
            else:
                result[key] = value if key != "info" else None
        return result


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
    """
    A decorator that transforms a SignalsDefinition object into
    a SignalGeneratorDefinition object.

    This function wraps another function and intercepts its first
    argument if it's a SignalsDefinition. It then transforms this
    SignalsDefinition into a SignalGeneratorDefinition with restructured
    operands and conditions.

    Parameters:
    func (Callable[..., Any]):
        The function to be wrapped. It should accept a SignalsDefinition
        as its first argument.

    Returns:
    Callable[..., Any]:
        A wrapped version of the input function that performs the
        transformation before calling the original function.

    The transformation process includes:
    1. Extracting operands from the conditions in the SignalsDefinition.
    2. Restructuring the conditions into open_long, close_long, open_short,
    and close_short categories.
    3. Replacing operand placeholders (a, b, c, d) with their actual names
    in the conditions.
    4. Creating a new SignalGeneratorDefinition with the transformed data.
    5. Replacing the original SignalsDefinition argument with the new
    SignalGeneratorDefinition.

    Note: This function modifies the arguments passed to the wrapped function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """TEST"""

        # Check if the first argument is a SignalsDefinition
        if args and isinstance(args[0], SignalsDefinition):
            signals_def = args[0]

            operands = {}
            conditions = {
                "open_long": [],
                "close_long": [],
                "open_short": [],
                "close_short": [],
            }

            for condition in signals_def.conditions:
                for i, operand in enumerate(["a", "b", "c", "d"]):
                    operand_attr = f"operand_{operand}"
                    if hasattr(condition, operand_attr):
                        operand_value = getattr(condition, operand_attr)
                        if operand_value:
                            operand_name = (
                                operand_value[0]
                                if isinstance(operand_value, tuple)
                                else operand_value
                            )
                            operands[operand_name] = operand_value

                for condition_type in [
                    "open_long",
                    "close_long",
                    "open_short",
                    "close_short",
                ]:
                    if hasattr(condition, condition_type):
                        condition_value = getattr(condition, condition_type)
                        if condition_value:
                            transformed_condition = list(condition_value)
                            for i, item in enumerate(transformed_condition):
                                if item in ["a", "b", "c", "d"]:
                                    operand_attr = f"operand_{item}"
                                    if hasattr(condition, operand_attr):
                                        operand_value = getattr(condition, operand_attr)
                                        operand_name = (
                                            operand_value[0]
                                            if isinstance(operand_value, tuple)
                                            else operand_value
                                        )
                                        transformed_condition[i] = operand_name
                            conditions[condition_type].append(
                                tuple(transformed_condition)
                            )

            transformed_def = SignalGeneratorDefinition(
                name=signals_def.name,
                operands=operands,
                conditions={k: v if v else None for k, v in conditions.items()},
            )
            # Replace the SignalsDefinition with the transformed
            # SignalGeneratorDefinition
            args = (transformed_def,) + args[1:]

        return func(*args, **kwargs)

    return wrapper


@transform_condition_definition
def dummy_test_function(signals_def: SignalsDefinition) -> None:
    print(f"Testing {signals_def.name}")
    print(signals_def.to_yaml())
    print("-" * 80)


if __name__ == "__main__":
    dummy_test_function(aroon_osc)
