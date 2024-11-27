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

from abc import abstractmethod
from plotly.io import to_image
from plotly.subplots import make_subplots

from .plot_definition import (
    PlotDefinition, SubPlot, Layout,
    Candlestick, Volume, Positions, Signal,
    Line, Channel, Drawdown
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

    def plot(self, data: pd.DataFrame | Data, p_def: PlotDefinition):
        """
        Plots a strategy and opens the image in a browser.

        Parameters
        ----------
        data: pd.DataFrame | tp.Data
            OHLCV data to plot, possibly with indicator and signal data

        p_def : PlotDefinition
            plot parameters for a strategy
        """
        config = {"autosizable": True, "responsive": True, "displaylogo": False}

        fig = self.create_fig(p_def)

        # draw subplots
        for subplot in p_def.subplots:
            subplot.draw_subplot(fig, data)

        fig.show(width=2400, height=1600, config=config)

        p_def.layout.show_layout()
        print(p_def)

    def create_fig(self, p_def: PlotDefinition):
        """Prepare a figure for plotting.

        Parameters
        ----------
        p_def : PlotDefinition
            plot parameters for a strategy
        """
        rows = p_def.layout.number_of_rows
        cols = p_def.layout.number_of_columns

        row_heights = [8, 5, 3]
        column_widths = tuple(1 for _ in range(cols))

        # secondary y axis for main plot
        specs = [
            [
                {"secondary_y": True if i == 0 and j == 0 else False}
                for j in range(cols)]
            for i in range(rows)
        ]

        if p_def.number_of_subplots == 0:
            fig = make_subplots(
                rows=rows,
                cols=1,
                specs=specs,
                subplot_titles=p_def.subplot_titles,
            )
        else:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                row_heights=row_heights,
                column_widths=column_widths,
                vertical_spacing=0.05,
                shared_xaxes=True,
                subplot_titles=p_def.subplot_titles,
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
        fig.update_xaxes(row=3, col=1, rangeslider_visible=False)
        fig.update_xaxes(row=rows, col=1, rangeslider_thickness=0.02)

        # set the font family, size, etc.
        font_color = self.style.colors.text.set_alpha(self.style.font_opacity).rgba

        logger.info("using font color: %s" % font_color)
        logger.info("font opacity is set to: %s" % self.style.font_opacity)
        logger.info("title font size is set to: %s" % self.style.title_font_size)

        fig.update_layout(
            font_family=self.style.font_family,
            font_color=font_color,
            font_size=self.style.font_size,
            title_font_family=self.style.font_family,
            title_font_color=font_color,
            title_font_size=self.style.title_font_size,
            legend_title_font_color=font_color,
            showlegend=False,
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
            margin=dict(l=50, r=50, t=40, b=30, pad=0),
            autosize=False,
        )

        # this is somehow necessary to avoid a mright side margin
        # that is larger than what was defined in the previous step
        fig.update_xaxes(domain=[0, 1])

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
            showgrid=True, gridwidth=0.5, gridcolor=color, zeroline=False,
            ticklen=20,  # showticklabels=False,
            )
        no_grid_params = dict(
            showgrid=False, zeroline=False, ticklen=10, showticklabels=True,
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

    def draw_ohlcv(self, fig: go.Figure, data: pd.DataFrame):
        vol_alpha = self.style.volume_opacity
        vol_color = self.style.colors.volume.set_alpha(vol_alpha).rgba
        border_color = self.style.colors.volume.set_alpha(vol_alpha / 2).rgba

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                showlegend=False,
                marker={
                    "color": vol_color,
                    "line": {"color": border_color, "width": 1}
                },
                zorder=-1
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                name="OHLC",
                line=dict(width=0.75),
                showlegend=False,
                opacity=1,
                zorder=0
            ),
            secondary_y=False,
        )

        return fig

    def _draw_signal(self, fig, data: pd.DataFrame):
        # if "buy_at" not in data.columns:
        #     data.loc[
        #         (data[signal_column] >= 1) & (data[signal_column] <= 20),
        #         "buy_at"
        #     ] = data.open

        # if "sell_at" not in data.columns:
        #     data.loc[
        #         (data[signal_column] <= -1) & (data[signal_column] >= -20),
        #         "sell_at"
        #     ] = data.open

        data.loc[data.open_long.shift() == 1, "open_long_at"] = data.open
        data.loc[data.open_short.shift() == 1, "open_short_at"] = data.open

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=data.open_long_at,
                marker=dict(
                    symbol="triangle-up",
                    color=self.style.colors.buy.rgba,
                    size=self.style.marker_size,
                    opacity=self.style.marker_opacity,
                    line=dict(color=self.style.colors.buy.rgba, width=1),
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=data.open_short_at,
                marker=dict(
                    symbol="triangle-down",
                    color=self.style.colors.sell.rgba,
                    size=self.style.marker_size,
                    opacity=self.style.marker_opacity,
                    line=dict(color=self.style.colors.sell.rgba, width=1),
                ),
                showlegend=False,
            )
        )

        return fig

    def draw_buys_and_sells(self, fig, data: pd.DataFrame):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=data.buy_at,
                marker=dict(
                    symbol="triangle-up",
                    color=self.style.colors.buy.rgba,
                    size=self.style.marker_size,
                    opacity=self.style.marker_opacity,
                    line=dict(color=self.style.colors.buy.rgba, width=1),
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=data.sell_at,
                marker=dict(
                    symbol="triangle-down",
                    color=self.style.colors.sell.rgba,
                    size=self.style.marker_size,
                    opacity=self.style.marker_opacity,
                    line=dict(color=self.style.colors.sell.rgba, width=1),
                ),
                showlegend=False,
            )
        )

        return fig

    def draw_positions(self, fig: go.Figure, data: pd.DataFrame, row=1, col=1):
        # Ensure the dataframe is sorted by index
        data = data.sort_index()

        # Get the 'position' column
        positions = data["position"]

        # Identify segments where 'position' remains the same
        segments = positions.ne(positions.shift()).cumsum()

        # Group the dataframe by these segments
        grouped = data.groupby(segments)

        # Define colors for long and short positions
        bull_color = "rgba(0, 125, 0, 0.1)"  # Green with low opacity
        bear_color = "rgba(125, 0, 0, 0.1)"  # Red with low opacity

        # Iterate over each group to draw rectangles
        for _, group in grouped:
            pos_value = group["position"].iloc[0]
            if pos_value != 0:
                fig.add_shape(
                    type="rect",
                    x0=group.index[0],
                    x1=group.index[-1],
                    y0=0,  # Use 0 for bottom of subplot
                    y1=1,  # Use 1 for top of subplot
                    xref=f"x{row}" if row > 1 else "x",
                    yref=f"y{row} domain" if row > 1 else "y domain",
                    fillcolor=bull_color if pos_value == 1 else bear_color,
                    layer="below",
                    line_width=0,
                    row=row,
                    col=col,
                )

        return fig

    # --------------------------------------------------------------------------------
    def draw_main_plot(self, fig, data: pd.DataFrame, p_def: PlotDefinition):
        for row in range(1, p_def.number_of_rows + 1):
            self.draw_positions(fig, data, row=row)

        self.draw_ohlcv(fig, data)
        self.draw_buys_and_sells(fig, data)

    def draw_subplot(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        desc: SubPlot,
        row: int
    ) -> go.Figure:
        logger.debug(f"drawing subplot {desc.label} in row {row}")

        default_line_def = Line(
            label=desc.label,
            color=self.style.colors.strategy.rgba,
            width=0,
            opacity=0,
            shape="spline",
        )

        for channel in desc.channel:
            # convert tuples to Channel objects
            if isinstance(desc.channel, tuple):
                desc.channel = Channel(
                    label=f"{desc.channel[0]} - {desc.channel[1]}",
                    upper=default_line_def,
                    lower=default_line_def,
                )

            logger.debug("drawing channel: {0}".format(channel))
            try:
                channel.add_trace(fig=fig, data=data, row=row + 1)
            except Exception as e:
                logger.error(channel)
                logger.error(
                    "unable to draw channel: '%s' -> %s"
                    % (channel.label, e),
                    exc_info=True
                    )

        for line in desc.lines:
            logger.debug("drawing line: {0} in row {1}".format(line, row))
            line = Line(line[0]) if isinstance(line, tuple) else line
            line.add_trace(fig=fig, data=data, row=row + 1)

        for trig in desc.triggers:
            logger.debug("drawing line: {0}".format(trig))
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=data.index,
                    y=data[trig[0]],
                    line=dict(color="blue", width=0.5),
                    opacity=0.3,
                ),
                row=row + 1,
                col=1,
            )

        if desc.secondary_y_axis:
            _, yaxis_key = self.get_axis_keys(fig, row, 1)
            axis_key_number = int(yaxis_key[-1])

            # Configure axes for the second subplot
            fig.update_layout({
                yaxis_key: dict(
                    title="desc.label", anchor="x", domain=[0, 0.5]
                    ),
                f'yaxis{axis_key_number + 1}': dict(
                    overlaying=f'y{axis_key_number}', side='right', showline=True
                    )
            })

        return fig

    # --------------------------------------------------------------------------------
    from plotly.subplots import make_subplots

    def get_axis_keys(self, fig, row, col):
        """
        Retrieve the x-axis and y-axis keys for a specific subplot.

        Parameters:
            fig: The Plotly figure object created using make_subplots.
            row: The row number of the desired subplot (1-based index).
            col: The column number of the desired subplot (1-based index).

        Returns:
            A tuple containing the x-axis key and y-axis key as strings.
        """
        # Get the subplot grid dimensions
        rows = fig._grid_ref[0]  # Total number of rows
        cols = fig._grid_ref[1]  # Total number of columns

        for elem, i in enumerate(fig._grid_ref):
            logger.debug('-' * 120)
            logger.debug("elem [%s]: %s" % (i, elem))

        # Calculate the subplot number (1-based index)
        subplot_number = (row - 1) * cols + col

        # Determine axis keys
        xaxis_key = f'xaxis{"" if subplot_number == 1 else subplot_number}'
        yaxis_key = f'yaxis{"" if subplot_number == 1 else subplot_number}'

        return xaxis_key, yaxis_key


# ================================= Chart classes ====================================
class Chart:
    """Base class for all plotly-based charts."""

    default_format: str = "png"
    default_width: int = 2400
    default_height: int = 1600
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
        img_bytes = to_image(
            fig=self.artist.build_figure(data=self.data, p_def=self.plot_definition),
            format=format,
            scale=scale,
            width=width,
            height=height
        )

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

    # ........................ Define Portfolio Components ...........................
    def _equity_channel(self):
        return Channel(
            label="Equity channel",
            upper=Line(
                label=config.strategy.equity.label,
                column=config.strategy.equity.column,
                color=self.style.colors.strategy,
                width=self.style.line_width + 1,
                zorder=0,
            ),
            lower=Line(
                label=config.capital.equity.label,
                column=config.capital.equity.column,
                color=self.style.colors.capital.set_alpha(
                    self.style.colors.capital.a / 2
                ),
                width=self.style.line_width + 1,
                fillmethod="tonexty",
                zorder=0,
            ),
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
                (0, self.style.colors.hodl_fill.set_alpha(self.style.colors.hodl.a).rgba),
                (0.1, self.style.colors.hodl_fill.set_alpha(self.style.colors.hodl.a).rgba),
                (0.8, self.style.colors.hodl_fill.reset().rgba),
                (1, self.style.colors.hodl_fill.reset().rgba),
            ]
        )

    def _strategy_drawdown(self):
        return Drawdown(
            label=config.strategy.drawdown.label,
            column=config.strategy.drawdown.column,
            color=self.style.colors.strategy,
            fillcolor=self.style.colors.strategy_fill,
            gradient=False,
            critical=-25,
        )

    def _capital_drawdown(self):
        return Drawdown(
            label=config.capital.drawdown.label,
            column=config.capital.drawdown.column,
            color=self.style.colors.capital,
            fillcolor=self.style.colors.capital_fill,
            gradient=False,
            critical=-15,
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
            {
                "OHLCV": {"row": 1, "col": 1, "secondary_y": True},
                "Portfolio": {"row": 2, "col": 1},
                "Drawdown": {"row": 3, "col": 1},
            }
        )
        self.layout.show_layout()
        self._plot_definition = self._build_plot_definition()

    def draw(self):
        self.artist.plot(data=self.data, p_def=self.plot_definition)

    def subplot_ohlcv(self):
        return SubPlot(
            label="OHLCV",
            elements=(
                Candlestick(),
                # Volume(label='Volume', column=config.ohlcv.volume),
                Positions(column=config.ohlcv.volume, opacity=self.style.fill_alpha),
            ),
            secondary_y=True,
        )

    def _build_plot_definition(self):
        return PlotDefinition(
            name=self.title or "Tikr Chart",
            subplots=(
                self.subplot_ohlcv(),
                self.subplot_portfolio(),
                self.subplot_drawdown(),
            ),
            layout=self.layout,
            style=self.style,
        )


if __name__ == "__main__":
    print(config.equity)
