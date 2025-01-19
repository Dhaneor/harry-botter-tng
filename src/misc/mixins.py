#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 00:44:23 2025

@author dhaneor
"""
from analysis.chart.plot_definition import SubPlot, PlotDefinition
from analysis.chart.plotly_styles import backtest_style
from analysis.chart.tikr_charts import ChartArtist

class PlottingMixin:

    artist = ChartArtist(backtest_style)

    @property
    def plot_definition(self):
        self._update_layout()
        self.plot_layout.show_layout()

        # SubPlot classes can be added with each other, which is 
        # necessary if multiple elements (e.g.: moving averages) 
        # are part of the main plot, which in most cases is the 
        # OHLCV data
        main = sum([sp for sp in self.subplots if not sp.is_subplot])

        # all other indicators/operands need to be plotted in a 
        # separate subplot
        subs = [sp for sp in self.subplots if sp.is_subplot]

        # combine all subplots into a single list
        subplots = [main, *subs] if main else subs

        for subplot in subplots:
            for elem in subplot.elements:
                elem.color = self.artist.style.colors.strategy

        return PlotDefinition(
            title=self.name or "Anonymous Chart",
            subplots=subplots,
            layout=self.plot_layout,
            style=backtest_style,
        )   
    
    def plot(self):
        # Check if necessary attributes are present. Doing this here
        # makes an __init__ method unnecessary and classes which use 
        # this mixin do not need to call super().__init__().
        if not hasattr(self, 'subplots'):
            raise AttributeError(
                f"{self.__class__.__name__} must have a 'subplots' attribute"
                )

        for subplot in self.subplots:
            if not isinstance(subplot, SubPlot):
                raise TypeError("Each subplot must be an instance of 'SubPlot'")
            
        if not hasattr(self, 'plot_layout'):
            raise AttributeError(
                f"{self.__class__.__name__} must have a 'plot_layout' attribute"
                )
        
        # if not hasattr(self, 'plot_data'):
        #     raise AttributeError(
        #         f"{self.__class__.__name__} must have a 'plot_data' attribute"
        #         )

        self.artist.plot(data=self.plot_data, p_def=self.plot_definition)
    
    def _update_layout(self):
        for subplot in self.subplots:
            row = self.plot_layout.number_of_rows + 1
            if subplot.is_subplot:
                print(f"Adding subplot {subplot.label} at row {row}")
                self.plot_layout.layout[subplot.label] = {"row": row, "col": 1}
                self.plot_layout.row_heights.append(1)
            else:
                self.plot_layout.layout[subplot.label] = {"row": 1, "col": 1}
                if self.plot_layout.row_heights != 8:
                    self.plot_layout.row_heights.insert(0, 8)

        # self.plot_layout.show_layout()

     