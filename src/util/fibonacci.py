#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 00:22:20 2022

@author dhaneor
"""
from numba import jit, int64
from typing import List

@jit(nopython=True)
def fibonacci(n: int) -> int:
    """Calculate the n-th nummber in the Fibonacci series.

    :param n: n-th position in Fibonacci Series
    :type n: int
    :raises ValueError: ValueError if n is smaller than 0
    :return: the n-th number in the Fibonacci Series
    :rtype: int
    """
    if n < 0:
        raise ValueError(f'n cannot be smaller than zero!')
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

@jit(nopython=True, cache=True)
def fibonacci_series(up_to: int) -> set:
    """Gets the Fibonacci series up to a given number.

    :param up_to: upper bound for the series, 
    :type up_to: int
    :return: _description_
    :rtype: set
    """
    number_in_series, fib = 1, 0
    series: List[int64] = list() # type: ignore
    
    while fib < up_to:
        fib = fibonacci(number_in_series)
        
        if fib < up_to:
            series.append(fib)
            
        number_in_series += 1
                
    return set(series) # tuple(sorted(set(series)))
    
# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    
    for _ in range(5):
        print(fibonacci_series(10000))
    