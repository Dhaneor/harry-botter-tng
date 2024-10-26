#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides an optimizer class for strategy parameters.

Created on July 06 21:12:20 2023

@author dhaneor
"""
import itertools
import logging
from typing import Sequence, Iterable, Generator, Tuple, Any

# from scipy.optimize import minimize
# from scipy.optimize import Bounds
# from scipy.optimize import LinearConstraint
# from scipy.optimize import NonlinearConstraint

logger = logging.getLogger('main.optimizer')
logger.setLevel(logging.DEBUG)

# Type aliases
Combinations = Generator[Tuple[Any, ...], None, None]


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


def generate_combinations(*iterables: Iterable) -> Combinations:
    """Generates all possible combinations of elements from the given iterables.

    Args:
        *iterables: A variable number of iterable objects.

    Yields:
        A tuple representing a combination of elements, one from each iterable.
    """
    if not iterables:
        yield ()
    else:
        first, *rest = iterables
        for item in first:
            for combination in generate_combinations(*rest):
                yield (item,) + combination


# ======================================================================================
if __name__ == '__main__':
    a = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    b = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    c = (True, False)
    d = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

    i = 0
    for _ in generate_combinations(a, b, c, d):
        if i == 0:
            print("First combination:", _)
        i += 1

    print(f"Generated {i} combinations.")


