#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import base64
import io
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import time

from abc import abstractmethod
from plotly.io import to_image
from plotly.subplots import make_subplots
from typing import Sequence

from .plot_definition import (
    PlotDefinition,
    SubPlot,
    Layout,
    Candlestick,
    Volume,
    Positions,
    Buy,
    Sell,
    Line,
    Channel,
    Drawdown,
    Leverage,
    PositionSize,
    Signals
)
from .plotly_styles import styles, TikrStyle
from .util import config

logger = logging.getLogger(f"main.{__name__}")

Data = dict[str, npt.NDArray[np.flexible]]

STYLES = styles


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

    def build_figure(self, data: pd.DataFrame | Data, p_def: PlotDefinition):
        """
        Plots a strategy and opens the image in a browser.

        Parameters
        ----------
        data: pd.DataFrame | tp.Data
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

    def plot(self, data: pd.DataFrame | Data, p_def: PlotDefinition) -> None:
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


# ================================= Chart classes ====================================
class Chart:
    """Base class for all plotly-based charts."""

    default_format: str = "png"
    default_width: int = 2400
    default_height: int = 1200
    default_scale: int = 1

    def __init__(self, data, style, title=None):
        self.data = self._prepare_data(data.copy())
        self.title = title

        self.style: TikrStyle
        self._set_style(style)

        self._plotdefinition: PlotDefinition = None

        self.artist = ChartArtist(self.style)

    @property
    def plot_definition(self):
        return self._build_plot_definition()

    @abstractmethod
    def draw(self): ...

    def get_image_bytes(
        self,
        format: str = default_format,
        width: int = default_width,
        height: int = default_height,
        scale: int = default_scale,
    ) -> io.BytesIO:
        """
        Generate a high-quality BytesIO object containing the chart image.

        Args:
            format (str): Image format ('png', 'jpeg', 'webp', 'svg', 'pdf')
            scale (int): Scale factor for the image (default 3 for high quality)
            width (int): Base width of the image in pixels (default 2400)
            height (int): Base height of the image in pixels (default 1600)

        Returns:
            io.BytesIO: BytesIO object containing the high-quality image
        """
        start_time = time.time()

        img_bytes = to_image(
            fig=self.artist.build_figure(data=self.data, p_def=self.plot_definition),
            format=format,
            scale=scale,
            width=width,
            height=height,
        )

        execution_time = round(time.time() - start_time)
        logger.info(f"Chart image generation took {execution_time} seconds.")

        return io.BytesIO(img_bytes)

    # .............................. Helper Methods ..................................
    @abstractmethod
    def _build_plot_definition(self): ...

    def _prepare_data(self, data):
        data = pd.DataFrame.from_dict(data) if isinstance(data, dict) else data

        for col in data.columns:
            data[col] = data[col] * -100 if "drawdown" in col else data[col]

        return data

    def _set_style(self, style: str):
        if style not in STYLES:
            logger.warning(f"Style '{style}' not found. Using default style.")
            style = "default"

        self.style = STYLES[style]

    # ......................... Define Position Components ...........................
    def subplot_position_size(self) -> SubPlot:
        return SubPlot(
            label="Position size",
            elements=[
                PositionSize(
                    label=config.position.size.label,
                    column=config.position.size.column,
                    color=self.style.colors.strategy,
                )
            ],
        )

    def subplot_leverage(self) -> SubPlot:
        return SubPlot(
            label="Leverage",
            elements=[
                Leverage(
                    label=config.position.leverage.label,
                    column=config.position.leverage.column,
                )
            ],
        )

    # ........................ Define Portfolio Components ...........................
    def _equity_channel(self):
        return Channel(
            label="Equity channel",
            upper=Line(
                label=config.strategy.equity.label,
                column=config.strategy.equity.column,
                color=self.style.colors.strategy,
                width=self.style.line_width + 1,
                zorder=1,
            ),
            lower=Line(
                label=config.capital.equity.label,
                column=config.capital.equity.column,
                color=self.style.colors.capital.set_alpha(
                    self.style.colors.capital.a / 2
                ),
                width=self.style.line_width + 1,
                zorder=1,
            ),
            fillmethod="tonexty",
            color=self.style.colors.strategy_fill,
        )

    def _hodl_equity(self):
        return Line(
            label=config.hodl.equity.label,
            column=config.hodl.equity.column,
            color=self.style.colors.hodl,
            zorder=-1,
        )

    # ......................... Define Drawdown Components ...........................
    def _hodl_drawdown(self):
        return Drawdown(
            label=config.hodl.drawdown.label,
            column=config.hodl.drawdown.column,
            color=self.style.colors.hodl,
            fillcolor=self.style.colors.hodl_fill,  # fill_color
            opacity=0.1,
            color_scale=[
                (
                    0,
                    self.style.colors.hodl_fill.set_alpha(
                        self.style.colors.hodl.a
                    ).rgba,
                ),
                (
                    0.1,
                    self.style.colors.hodl_fill.set_alpha(
                        self.style.colors.hodl.a
                    ).rgba,
                ),
                (0.8, self.style.colors.hodl_fill.reset().rgba),
                (1, self.style.colors.hodl_fill.reset().rgba),
            ],
        )

    def _strategy_drawdown(self):
        return Drawdown(
            label=config.strategy.drawdown.label,
            column=config.strategy.drawdown.column,
            color=self.style.colors.strategy,
            fillcolor=self.style.colors.strategy_fill,
            gradient=False,
            critical=-20,
        )

    def _capital_drawdown(self):
        return Drawdown(
            label=config.capital.drawdown.label,
            column=config.capital.drawdown.column,
            color=self.style.colors.capital,
            fillcolor=self.style.colors.capital_fill,
            gradient=False,
            critical=-10,
        )

    # ........................... (Sub-)Plot Descriptions ............................
    def subplot_portfolio(self) -> SubPlot:
        return SubPlot(
            label="Portfolio",
            elements=[
                self._equity_channel(),
                self._hodl_equity(),
            ],
        )

    def subplot_drawdown(self) -> SubPlot:
        return SubPlot(
            label="Drawdown",
            elements=[
                self._hodl_drawdown(),
                self._strategy_drawdown(),
                self._capital_drawdown(),
            ],
        )


class TikrChart(Chart):
    def __init__(self, df, style: str, title=None):
        super().__init__(df, style, title)

        self.layout = Layout(
            layout={
                "OHLCV": {"row": 1, "col": 1, "secondary_y": True},
                "Portfolio": {"row": 2, "col": 1},
                "Drawdown": {"row": 3, "col": 1},
            },
            row_heights=[8, 5, 3],
            col_widths=[1]
        )
        self.layout.show_layout()
        self._plot_definition = self._build_plot_definition()
        self.artist.plot_definition = self._plot_definition

    def draw(self):
        self.artist.plot(data=self.data, p_def=self.plot_definition)

    def subplot_ohlcv(self):
        return SubPlot(
            label="OHLCV",
            elements=(
                Positions(column=config.ohlcv.volume, opacity=self.style.fill_alpha),
                Candlestick(),
                Buy(),
                Sell(),
            ),
            secondary_y=True,
        )

    def _build_plot_definition(self):
        return PlotDefinition(
            title=self.title or "Tikr Chart",
            subplots=(
                self.subplot_ohlcv(),
                self.subplot_portfolio(),
                self.subplot_drawdown(),
            ),
            layout=self.layout,
            style=self.style,
        )


class BacktestChart(Chart):
    def __init__(self, df, style: str, title=None):
        logger.error(df.index.name)
        super().__init__(df, style, title)

        self.layout = Layout(
            layout={
                "OHLCV": {"row": 1, "col": 1, "secondary_y": True},
                "Portfolio": {"row": 2, "col": 1},
                "Drawdown": {"row": 3, "col": 1},
                "Leverage": {"row": 4, "col": 1},
                "Signal": {"row": 5, "col": 1},
            },
            row_heights=[12, 7, 5, 3, 2],
            col_widths=[1]
        )

        if logger.level == logging.DEBUG:
            self.layout.show_layout()

        self._plot_definition = self._build_plot_definition()
        self.artist.plot_definition = self._plot_definition

    def draw(self):
        self.artist.plot(data=self.data, p_def=self.plot_definition)

    def show(self) -> None:
        self.draw()

    # ......................... (Sub-)Plot Descriptions ..............................
    def subplot_signals(self):
        return SubPlot(
            label="Signal",
            elements=[
                Signals(
                    label=config.signals.label,
                    column=config.signals.column
                )
            ],
        )

    def _equity_channel(self):
        return Channel(
            label="Equity channel",
            upper=Line(
                label=config.strategy.equity.label,
                column=config.strategy.equity.column,
                color=self.style.colors.strategy,
                width=self.style.line_width,
                zorder=1,
            ),
            lower=Line(
                label=config.capital.equity.label,
                column=config.capital.equity.column,
                color=self.style.colors.capital.set_alpha(
                    self.style.colors.capital.a / 2
                ),
                width=self.style.line_width,
                zorder=1,
            ),
            fillmethod="tonexty",
            color=self.style.colors.strategy_fill,
        )

    def subplot_portfolio(self) -> SubPlot:
        return SubPlot(
            label="Portfolio",
            elements=[
                self._equity_channel(),
                self._hodl_equity(),
            ],
        )

    def subplot_ohlcv(self):
        return SubPlot(
            label="OHLCV",
            elements=(
                Positions(column=config.ohlcv.volume, opacity=self.style.fill_alpha),
                Volume(label='Volume', column=config.ohlcv.volume),
                Candlestick(),
                Buy(),
                Sell(),
            ),
            secondary_y=True,
        )

    def _build_plot_definition(self):
        return PlotDefinition(
            title=self.title or "Tikr Chart",
            subplots=(
                self.subplot_ohlcv(),
                self.subplot_portfolio(),
                self.subplot_drawdown(),
                self.subplot_leverage(),
                self.subplot_signals(),
            ),
            layout=self.layout,
            style=self.style,
        )


class SignalChart(Chart):
    indicator_row_height = 2

    def __init__(
        self,
        data: pd.DataFrame,
        subplots: Sequence[SubPlot],
        style: str,
        title: str = None
    ):
        super().__init__(data, style, title)

        self.subplots = subplots

        self.layout = Layout(
            layout={
                "OHLCV": {"row": 1, "col": 1, "secondary_y": True},
            },
            row_heights=[8],
            col_widths=[1]
        )
        self._update_layout()
        self.layout.show_layout()

        # self.clean_signals()
        self.add_buy_at_column()
        self.add_sell_at_column()

        self.style.line_width = 0.5

        print(self.data.tail(50))

        self._plot_definition = self._build_plot_definition()
        self.artist.plot_definition = self._plot_definition

    @property
    def ohlcv_elements(self):
        return [sub for sub in self.subplots if not sub.is_subplot]

    def draw(self):
        self.artist.plot(data=self.data, p_def=self.plot_definition)

    def subplot_ohlcv(self):
        main_indicators = self.ohlcv_elements
        traces = []
        for indicator in main_indicators:
            for trace in indicator.elements:
                trace.color = self.style.colors.strategy
                traces.append(trace)

        return SubPlot(
            label="OHLCV",
            elements=(
                # Positions(column=config.ohlcv.volume, opacity=self.style.fill_alpha),
                Volume(label='Volume', column=config.ohlcv.volume),
                *traces,
                Candlestick(),
                Buy(),
                Sell(),
            ),
            secondary_y=True,
        )

    def _update_layout(self):
        row = len(self.layout.layout)
        for subplot in self.subplots:
            if subplot.is_subplot:
                row += 1
                self.layout.layout[subplot.label] = {"row": row, "col": 1}
                self.layout.row_heights.append(self.indicator_row_height)

    def _build_plot_definition(self):
        subplots = [sp for sp in self.subplots if sp.is_subplot]

        for subplot in subplots:
            for elem in subplot.elements:
                elem.color = self.style.colors.strategy

        return PlotDefinition(
            title=self.title or "Tikr Chart",
            subplots=(
                self.subplot_ohlcv(),
                *subplots,
            ),
            layout=self.layout,
            style=self.style,
        )

    def clean_signals(self) -> None:
        """
        Clean the signals by removing consecutive repetitions and
        keeping only the first occurrence.
        """
        for column in ('open_long', 'close_long', 'open_short', 'close_short'):
            self.data[column] = self.data[column].replace(0, np.nan)
            signals = self.data[column].values
            cleaned_signals = np.full_like(signals, np.nan)
            last_signal = np.nan
            last_signal_arr = np.full_like(signals, np.nan)

            for i, signal in enumerate(signals):
                if not np.isnan(signal) and signal != last_signal:
                    cleaned_signals[i] = signal
                    last_signal = signal
                last_signal_arr[i] = last_signal

            self.data[column] = cleaned_signals
            self.data[f'last_{column}'] = last_signal_arr

    def add_buy_at_column(self) -> None:
        """
        Add a 'buy_at' column to the DataFrame where:
        a) the value is the close price when open_long is not NaN
        b) the value is np.nan for all other rows
        """
        self.data['buy_at'] = np.nan
        self.data['open_long'] = self.data.open_long.replace(0, np.nan)

        self.data.loc[
            (self.data['open_long'] > 0) | (self.data['close_short'] > 0),
            'buy_at'
        ] = self.data.close

    def add_sell_at_column(self) -> None:
        """
        Add a 'sell_at' column to the DataFrame where:
        a) the value is the close price when close_long is not NaN
        b) the value is np.nan for all other rows
        """
        self.data['sell_at'] = np.nan
        self.data['open_short'] = self.data.open_short.replace(0, np.nan)

        self.data.loc[
            (self.data['open_short'] > 0) | (self.data['close_long'] > 0),
            'sell_at'
        ] = self.data.close
