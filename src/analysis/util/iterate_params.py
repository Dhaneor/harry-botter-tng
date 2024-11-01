#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a factory for different indicators for Technical Analysis.

Created on Mon Oct 28 20:03:50 2024

@author: dhaneor
"""
from itertools import product
from typing import TypeVar, Generator
from collections.abc import Iterable

T = TypeVar("T")


def vector_generator(
    iterables: list[Iterable[float | int]]
) -> Generator[list[T], None, None]:
    """
    Generator that yields all possible combinations of values from the given iterables.

    Parameters:
    ----------
    iterables : list[Iterable[float | int]]
        List of iterables, each containing a sequence of values.

    Yields:
    -------
    list[T]
        A combination of values, one from each of the input iterables.
    """
    for combination in product(*iterables):
        yield list(combination)


def vector_diff(vector1: list[T], vector2: list[T]) -> list[T]:
    """
    Difference between two vectors.

    Parameters:
    ----------
    vector1 : list[T]
    First vector.
    vector2 : list[T]
    Second vector.
    Returns:
    list[T]
    Difference between vector1 and vector2.
    """
    return [v1 - v2 for v1, v2 in zip(vector1, vector2)]


if __name__ == "__main__":
    v1 = (i for i in range(10))
    v2 = (i for i in range(10, 20))
    v3 = (i for i in range(30, 40))

    for combination in vector_generator((v1, v2, v3)):
        print(combination)
