#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides an optimizer class for strategy parameters.

Created on July 06 21:12:20 2023

@author dhaneor
"""
import itertools
import logging
from typing import Sequence

# from scipy.optimize import minimize
# from scipy.optimize import Bounds
# from scipy.optimize import LinearConstraint
# from scipy.optimize import NonlinearConstraint

logger = logging.getLogger('main.optimizer')
logger.setLevel(logging.DEBUG)


def generate_param_combinations(
    parameter_spaces: Sequence[dict[Sequence]]
) -> list[tuple]:
    """
    Generate combinations of parameter values.

    Parameters
    ----------
    parameter_spaces
        A sequence of dictionaries representing parameter spaces.
        Each dictionary should have keys 'mins', 'maxs', and 'steps',
        with corresponding values as sequences of integers.

    Returns
    -------
    List[Tuple]
        A list of tuples representing all possible combinations of
        parameter values from the given parameter spaces.
    """
    logger.debug(
        f"generating combinations for {len(parameter_spaces)} parameter spaces"
    )

    combination_sets = []

    for space in parameter_spaces:
        for value in space.values():
            logger.debug(f"generating combinations for {value}")
            if len(value) == 2:
                value.append(1)
            mins, maxs, steps = value
            vals = range(mins, maxs + 1, steps)
            combination_sets.append(vals)

    return list(itertools.product(*combination_sets))


# ======================================================================================
if __name__ == '__main__':
    parameter_spaces = [
        {'timeperiod': [2, 100, 2]},
        {'trigger': [5, 35, 1]},
        {'trigger': [65, 95, 1]},
    ]

    parameter_spaces = [
        {'timeperiod': [2, 200, 5]},
        {'trigger': [70, 200, 10]},
        {'trigger': [70, 200, 10]},
        {'timeperiod': [2, 200, 5]},
        {'trigger': [5, 35, 5]},
        {'trigger': [65, 95, 5]}
    ]

    # parameter_spaces = [
    #     {'timeperiod': [2, 200, 1]},
    #     {'timeperiod': [2, 200, 1]},
    # ]

    length = 0

    for item in parameter_spaces:
        for k, v in item.items():
            if length == 0:
                length = len(range(*v))
            else:
                length = length * len(range(*v))

    print(f"{length:,.0f} parameter spaces")

    # sys.exit()

    combinations = generate_param_combinations(parameter_spaces)
    filtered = combinations  # list(filter(lambda x: x[1] < x[2], combinations))

    print(filtered[-10:])

    et = 0.0003  # approximate time for one backtest

    print(
        f"we have {len(filtered):,.0f}/{len(combinations):,.0f} combinations "
        f"-> estimated time for optimizer: {round(len(filtered) * et):,.0f} seconds"
    )
