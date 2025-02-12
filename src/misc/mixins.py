#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 00:44:23 2025

@author dhaneor
"""

import pandas as pd

from analysis.chart import Layout, SubPlot, PlotDefinition, Line, Signal, styles
from analysis.chart.chart_artist import ChartArtist

STYLE = styles["backtest"]


class PlottingMixin:

    style = STYLE
    artist = ChartArtist(STYLE)
    plot_layout = Layout(
        layout = dict(),
        row_heights = [],
        col_widths = [1],
    )
    main_height: int = 4

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
        
        for subplot in subplots:
            line_no = 0
            for elem in subplot.elements:
                if isinstance(elem, Line):
                    elem.color = line_colors[line_no] 
                    line_no += 1   
                elif not isinstance(elem, Signal):
                    elem.color = self.artist.style.colors.strategy

        return PlotDefinition(
            title=self.display_name or "Anonymous Chart",
            subplots=subplots,
            layout=self.plot_layout,
            style=self.style,
        )   
    
    def plot(self):
        for subplot in self.subplots:
            if not isinstance(subplot, SubPlot):
                raise TypeError(
                    "Each subplot must be an instance of 'SubPlot' "
                    f"got: {type(subplot)}"
                    )

        data = self.plot_data

        for k,v in data.items():
            print(f"shape of array for {k}: {v.shape}")
                
        data = pd.DataFrame.from_dict(data) if isinstance(data, dict) else data

        if "open time" in data.columns:
            data["open time"] = pd.to_datetime(
                (data["open time"] / 1000).astype(int),
                utc=False,
                origin="unix",
                unit="s",
                )
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
