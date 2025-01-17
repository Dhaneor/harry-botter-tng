#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 00:44:23 2025

@author dhaneor
"""

class PlottingMixin:
    def plot(self, **kwargs):
        if not hasattr(self, 'plot_definition'):
            raise AttributeError(f"{self.__class__.__name__} must have a plot_definition attribute")
        
        plot_def = self.plot_definition
        if not isinstance(plot_def, PlotDefinition):
            raise TypeError("plot_definition must be an instance of PlotDefinition")
        
        data = self.get_plot_data()  # This method should be implemented by each class
        strategy_plot(data, plot_def, **kwargs)

    def get_plot_data(self):
        raise NotImplementedError("Subclasses must implement get_plot_data()")