#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 19 16:19:23 2025

@author dhaneor
"""
import base64
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_definition import PlotDefinition, Candlestick
from .plotly_styles import TikrStyle

logger = logging.getLogger(f"main.{__name__}")



class ChartArtist:
    """Class to draw charts based on chart descriptions.

    This class is intended to be a component of the instances of the
    various chart classes (see below), but it can also be used outside
    of them.

    The chart descriptions are standardized in the PlotDefinition class
    (see plot_defintions.py) and are prepared by the  chart classes or
    by the SignalGenerator class from the analysis package.
    """

    signal_column: str = "signal"

    def __init__(self, style: TikrStyle):
        self.style = style
        self.plot_definition: PlotDefinition = None

    def build_figure(self, data: pd.DataFrame | dict[str, np.ndarray], p_def: PlotDefinition):
        """
        Plots a strategy and opens the image in a browser.

        Parameters
        ----------
        data: pd.DataFrame | tp.dict[str, np.ndarray]
            OHLCV data to plot, possibly with indicator and signal data

        p_def : PlotDefinition
            plot parameters for a strategy
        """
        self.plot_definition = p_def
        self.fig = self.create_fig(p_def)

        # draw subplots, including their legends
        for subplot_no, subplot in enumerate(p_def.subplots):
            subplot.draw_subplot(self.fig, data)

            # the legend names need to start from 'legend_2' to avoid
            # overwriting the main plot legend
            legend_name = f"legend{subplot_no + 2}"

            # update all traces in the current subplot with the legend name
            self.fig.update_traces(row=subplot.row, col=subplot.col, legend=legend_name)

            # add legend to subplot
            self.position_legend_in_subplot(
                legend_name=legend_name,
                row=subplot.row,
                col=subplot.col,
                legend_position='upper left'
            )

        return self.fig

    def plot(self, data: pd.DataFrame | dict[str, np.ndarray], p_def: PlotDefinition) -> None:
        self.build_figure(data, p_def)
        config = {"autosizable": True, "responsive": True, "displaylogo": False}
        self.fig.show(width=2400, height=1600, config=config)

    def create_fig(self, p_def: PlotDefinition):
        """Prepare a figure for plotting.

        Parameters
        ----------
        p_def : PlotDefinition
            plot parameters for a strategy
        """
        rows = self.plot_definition.layout.number_of_rows
        cols = self.plot_definition.layout.number_of_columns

        row_heights = self.plot_definition.layout.row_heights
        column_widths = self.plot_definition.layout.col_widths

        # secondary y axis for main plot
        specs = [
            [{"secondary_y": True if i == 0 and j == 0 else False} for j in range(cols)]
            for i in range(rows)
        ]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            row_heights=row_heights,
            column_widths=column_widths,
            vertical_spacing=self.plot_definition.layout.vertical_spacing,
            horizontal_spacing=self.plot_definition.layout.horizontal_spacing,
            shared_yaxes=True,
            shared_xaxes=True,
            subplot_titles=None,  # self.plot_definition.subplot_titles,
            specs=specs,
        )

        fig = self.add_gridlines(fig, p_def)

        style = self.style

        # set some basic style parameters and make the y axis logarithmic
        if isinstance(p_def.subplots[0], Candlestick):
            fig.update_layout(yaxis_type="log", hovermode="x")

        # set background colors
        fig.update_layout(
            plot_bgcolor=style.colors.background.rgba,
            paper_bgcolor=style.colors.canvas.rgba,
        )

        # remove the zeroline
        fig.update_layout(yaxis=dict(zeroline=False), yaxis2=dict(zeroline=False))

        # disable range sliders
        fig.update_xaxes(row=1, col=1, rangeslider_visible=False)
        fig.update_xaxes(row=2, col=1, rangeslider_visible=False)
        fig.update_xaxes(rangeslider_visible=False)
        # fig.update_xaxes(row=rows, col=1, rangeslider_thickness=0.02)

        # set the font family, size, etc.
        font_color = self.style.colors.text.set_alpha(self.style.font_opacity).rgba

        logger.debug("using font color: %s" % font_color)
        logger.debug("font opacity is set to: %s" % self.style.font_opacity)
        logger.debug("title font size is set to: %s" % self.style.title_font_size)

        fig.update_layout(
            font_family=self.style.font_family,
            font_color=font_color,
            font_size=self.style.font_size,
            title_font_family=self.style.font_family,
            title_font_color=font_color,
            title_font_size=self.style.title_font_size,
            legend_title_font_color=font_color,
            showlegend=True,
        )

        fig.update_annotations(font_size=self.style.font_size)

        # set tick font family, size and color
        tick_font_definition = dict(
            family=self.style.font_family,
            color=font_color,
            size=self.style.tick_font_size,
        )

        fig.update_yaxes(tickfont=tick_font_definition)
        fig.update_xaxes(tickfont=tick_font_definition)

        # adjust the margins
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=30, pad=0),
            autosize=False,
        )

        # this is somehow necessary to avoid a right side margin
        # that is larger than what was defined in the previous step
        fig.update_xaxes(domain=[0, 1])

        fig.update_layout(
            title_text=self.plot_definition.title,
            title_x=0.5,
            title_font_size=self.style.title_font_size,
            title_font_family=self.style.font_family,
            title_font_color=self.style.colors
            .text.set_alpha(self.style.font_opacity).rgba,
        )

        # set the background/overlay image, if defined in the style
        if self.style.canvas_image is not None:

            try:
                with open(self.style.canvas_image, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError as e:
                logger.error(f"Canvas image file not found: {e}")
            else:
                logger.info("Using canvas image: %s", self.style.canvas_image)
                fig.update_layout(
                    images=[
                        dict(
                            source=f"data:image/jpg;base64,{encoded_image}",
                            xref="paper",
                            yref="paper",
                            x=-0.05,
                            y=1.05,
                            sizex=1.1,
                            sizey=1.1,
                            xanchor="left",
                            yanchor="top",
                            layer="above",
                            sizing="stretch",
                            opacity=self.style.canvas_image_opacity,
                        )
                    ],
                )

        return fig

    def add_gridlines(self, fig: go.Figure, p_def: PlotDefinition):
        # add_gridlines(fig, color=self.style.colors.grid.rgba)
        # make sure the grid is drwan first for all subplots
        # Update grid for all subplots
        rows = p_def.layout.number_of_rows
        color = self.style.colors.grid.rgba
        update_params = dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor=color,
            zeroline=False,
            ticklen=20,  # showticklabels=False,
        )
        no_grid_params = dict(
            showgrid=False,
            zeroline=False,
            ticklen=10,
            showticklabels=True,
        )

        # Update x-axes for all rows
        for i in range(1, rows + 1):
            fig.update_xaxes(update_params, row=i, col=1)

        # Update y-axes for all rows
        for i in range(1, rows + 1):
            if i == 1:  # Main plot with two y-axes
                fig.update_yaxes(update_params, row=i, col=1, secondary_y=False)
                fig.update_yaxes(no_grid_params, row=i, col=1, secondary_y=True)
            else:  # Other subplots with single y-axis
                fig.update_yaxes(update_params, row=i, col=1)

        return fig

    # --------------------------------------------------------------------------------
    def get_subplot_domain(self, row, col):
        def normalize(values):
            total = sum(values)
            return [v / total for v in values]

        rows = self.plot_definition.layout.number_of_rows
        cols = self.plot_definition.layout.number_of_columns

        row_heights_normalized = normalize(self.plot_definition.layout.row_heights)
        column_widths_normalized = normalize(self.plot_definition.layout.col_widths)

        vertical_spacing = self.plot_definition.layout.vertical_spacing
        horizontal_spacing = self.plot_definition.layout.horizontal_spacing

        # Calculate total spacings
        total_vertical_spacing = vertical_spacing * (rows - 1)
        total_horizontal_spacing = horizontal_spacing * (cols - 1)

        # Adjust heights and widths to account for spacing
        adjusted_row_heights = [
            h * (1 - total_vertical_spacing) for h in row_heights_normalized
        ]
        adjusted_column_widths = [
            w * (1 - total_horizontal_spacing) for w in column_widths_normalized
        ]

        # Calculate cumulative positions
        y_positions = []
        y = 1.0  # Start from the top
        for idx, h in enumerate(adjusted_row_heights):
            y_start = y - h
            y_positions.append((y_start, y))
            if idx < rows - 1:
                y = y_start - vertical_spacing
            else:
                y = y_start

        x_positions = []
        x = 0.0  # Start from the left
        for idx, w in enumerate(adjusted_column_widths):
            x_end = x + w
            x_positions.append((x, x_end))
            if idx < cols - 1:
                x = x_end + horizontal_spacing
            else:
                x = x_end

        # Get the domain for the specific subplot
        x_start, x_end = x_positions[col - 1]
        y_start, y_end = y_positions[row - 1]  # Rows are indexed from top to bottom

        return (x_start, x_end), (y_start, y_end)

    def position_legend_in_subplot(
        self,
        legend_name,
        row=1,
        col=1,
        legend_position="upper left",
        xoffset=0.02,
        yoffset=0.02,
        legend_kwargs=None,
    ):
        # Get the subplot domain
        (x_start, x_end), (y_start, y_end) = self.get_subplot_domain(row, col)

        # Adjust xoffset and yoffset relative to the subplot size
        x_range = x_end - x_start
        y_range = y_end - y_start

        xoffset_abs = xoffset * x_range
        yoffset_abs = yoffset * y_range

        # Determine legend position within the subplot domain
        if legend_position == 'upper left':
            x = x_start + xoffset_abs
            y = y_end - yoffset_abs
            xanchor = 'left'
            yanchor = 'top'
        elif legend_position == 'upper right':
            x = x_end - xoffset_abs
            y = y_end - yoffset_abs
            xanchor = 'right'
            yanchor = 'top'
        elif legend_position == 'lower left':
            x = x_start + xoffset_abs
            y = y_start + yoffset_abs
            xanchor = 'left'
            yanchor = 'bottom'
        elif legend_position == 'lower right':
            x = x_end - xoffset_abs
            y = y_start + yoffset_abs
            xanchor = 'right'
            yanchor = 'bottom'
        else:
            # Default to upper left
            x = x_start + xoffset_abs
            y = y_end - yoffset_abs
            xanchor = 'left'
            yanchor = 'top'

        # Build the legend dictionary
        legend_dict = dict(
            x=x,
            y=y,
            xanchor=xanchor,
            yanchor=yanchor,
            bgcolor=self.style.colors.canvas.set_alpha(0.1).rgba,
            font=dict(size=self.style.tick_font_size),
            tracegroupgap=5
        )

        if legend_kwargs:
            legend_dict.update(legend_kwargs)

        # Update figure layout
        logger.debug("updating legend '%s' with %s", legend_name, legend_dict)
        self.fig.update_layout({legend_name: legend_dict})

    def determine_scale(data):
        """Determine whether to use linear or log scale based on data range."""
        data_range = np.ptp(data)
        if data_range == 0:
            return 'linear'
        log_range = np.log10(np.max(np.abs(data))) - np.log10(np.min(np.abs(data[data != 0])))
        return 'symlog' if log_range > 2 else 'linear'
