#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides an optimizer class for strategy parameters.

Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import operator
from abc import ABC, abstractmethod
from itertools import product
from functools import reduce

from . import strategy_backtest as bt
from .strategies.signal_generator import SignalGenerator

logger = logging.getLogger('main.optimizer')
logger.setLevel(logging.DEBUG)

TIME_FOR_ONE_BACKTEST = 10  # execution time for one backtest in milliseconds


"""
arguments / pre-requisites:
    ohlcv data: A DataFrame containing OHLCV data
    objective_function: A function that takes parameters as input and returns a scalar value
    signal generator: A SignalGenerator instance with a defined strategy
    backtest function: A function that takes a SignalGenerator and a data dictionary as input and returns a DataFrame

"""

class OptimizationStrategy(ABC):
    @abstractmethod
    def generate_candidates(self):
        pass

    @abstractmethod
    def evaluate_candidate(self, candidate):
        pass

    @abstractmethod
    def update_population(self, results):
        pass


class GridSearch(OptimizationStrategy):
    def __init__(self, indicator):
        self.indicator = indicator

    def generate_candidates(self):
        return self.indicator.parameter_combinations

    def evaluate_candidate(self, candidate):
        # Implement evaluation logic
        pass

    def update_population(self, results):
        # No-op for grid search
        pass


class GeneticAlgorithm(OptimizationStrategy):
    def __init__(self, indicator, population_size):
        self.indicator = indicator
        self.population_size = population_size
        self.population = self._initialize_population()

    def _initialize_population(self):
        # Generate initial random population
        pass

    def generate_candidates(self):
        return self.population

    def evaluate_candidate(self, candidate):
        # Implement evaluation logic
        pass

    def update_population(self, results):
        # Implement selection, crossover, and mutation
        pass


optimization_strategies = {
    'grid_search': GridSearch,
    'genetic_algorithm': GeneticAlgorithm
}


class Optimizer:
    def __init__(
        self,
        signal_generator: SignalGenerator,
        optimization_strategy: str
    ):
        self.generator = signal_generator
        self.optimization_strategy = optimization_strategies.get(optimization_strategy)

        if not self.optimization_strategy:
            raise ValueError(f'Invalid optimization strategy: {optimization_strategy}')

    def optimize(self, iterations):
        for _ in range(iterations):
            candidates = self.optimization_strategy.generate_candidates()
            results = [
                self.optimization_strategy.evaluate_candidate(c) for c in candidates
                ]
            self.optimization_strategy.update_population(results)
            # Store and compare results

    def estimate_combinations(self) -> int:
        indicators = self.generator.indicators
        total_combinations = 1

        for indicator in indicators:
            # Get the number of possible values for each parameter
            param_counts = [len(list(param)) for param in indicator.parameters]

            # Multiply the counts to get the number of combinations for this indicator
            indicator_combinations = reduce(operator.mul, param_counts, 1)

            # Multiply with the total
            total_combinations *= indicator_combinations

        return total_combinations

    def estimate_execution_time(self) -> float:
        # Implement time estimation logic
        return self.estimate_combinations() * TIME_FOR_ONE_BACKTEST / 1000


# ======================================================================================
if __name__ == '__main__':
    pass
