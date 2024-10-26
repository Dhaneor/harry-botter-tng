#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Sat Aug 05 22:39:50 2023

@author: dhaneor
"""
from typing import Iterable, Generator, Tuple, Any

Combinations = Generator[Tuple[Any, ...], None, None]


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
