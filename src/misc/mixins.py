#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 00:44:23 2025

@author dhaneor
"""
import numpy as np
import pandas as pd

from analysis.chart.plot_definition import Layout, SubPlot, PlotDefinition, Line
from analysis.chart.plotly_styles import backtest_style
from analysis.chart.chart_artist import ChartArtist

class PlottingMixin:

    artist = ChartArtist(backtest_style)
    plot_layout = Layout(
        layout = dict(),
        row_heights = [],
        col_widths = [1],
    )
    main_height: int = 3

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

        # TODO: This is a hacky solution ... improve later!
        line_colors = [
            self.artist.style.colors.strategy,
            self.artist.style.colors.hodl,
            self.artist.style.colors.capital,
            self.artist.style.colors.candle_up,
            self.artist.style.colors.candle_down,
            self.artist.style.colors.buy,
            self.artist.style.colors.sell,
            self.artist.style.colors.volume,
        ]
        
        line_no = 0
        for subplot in subplots:
            for elem in subplot.elements:
                if isinstance(elem, Line):
                    print(f"Coloring line {line_no} with {line_colors[line_no]}")
                    elem.color = line_colors[line_no] 
                    line_no += 1   
                else:
                    elem.color = self.artist.style.colors.strategy

        return PlotDefinition(
            title=self.display_name or "Anonymous Chart",
            subplots=subplots,
            layout=self.plot_layout,
            style=backtest_style,
        )   
    
    def plot(self):
        # Check if necessary attributes are present. Doing this here
        # makes an __init__ method unnecessary and classes which use 
        # this mixin do not need to call super().__init__().
        # if not hasattr(self, 'subplots'):
        #     raise AttributeError(
        #         f"{self.__class__.__name__} must have a 'subplots' attribute"
        #         )    

        for subplot in self.subplots:
            if not isinstance(subplot, SubPlot):
                raise TypeError(
                    "Each subplot must be an instance of 'SubPlot' "
                    f"got: {type(subplot)}"
                    )
            
        if not hasattr(self, 'plot_layout'):
            raise AttributeError(
                f"{self.__class__.__name__} must have a 'plot_layout' attribute"
                )
        
        # if not hasattr(self, 'plot_data'):
        #     raise AttributeError(
        #         f"{self.__class__.__name__} must have a 'plot_data' attribute"
        #         )

        data = self.plot_data

        for k,v in data.items():
            print(f"shape of array for {k}: {v.shape}")
        
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        
        data.replace(np.nan, 0, inplace=True)

        if "open time" in data.columns:
            data["open time"] = pd.to_datetime(
                data["open time"],
                utc=True,
                # format="%Y-%m-%d %H:%M:%S.%f",
                errors="coerce",
                unit="ms")
            data.set_index("open time", inplace=True)

        print(data.tail(25))
        
        self.artist.plot(data=data, p_def=self.plot_definition)
    
    def _update_layout(self):
        for subplot in self.subplots:
            row = self.plot_layout.number_of_rows + 1
            if subplot.is_subplot:
                print(f"Adding subplot {subplot.label} at row {row}")
                self.plot_layout.layout[subplot.label] = {"row": row, "col": 1}
                self.plot_layout.row_heights.append(2)
            else:
                self.plot_layout.layout[subplot.label] = {"row": 1, "col": 1}
                if self.plot_layout.row_heights:
                    if self.plot_layout.row_heights[0] != self.main_height:
                        self.plot_layout.row_heights.insert(0, self.main_height)
                else:
                    self.plot_layout.row_heights.append(self.main_height)

        # self.plot_layout.show_layout()

     