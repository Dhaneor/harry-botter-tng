#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides tools to plot a strategy.

classes:
    PlotDescription
        plot parameters for one Condition

    PlotParamsStrat
        plot parameters for a strategy

Functions:
    plot
        plot function for one strategy

Created on Sat Aug 05 22:39:50 2023

@author: dhaneor
"""
import logging
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from plotly.subplots import make_subplots
from typing import NamedTuple, Sequence, Optional

from ..indicators.indicator import PlotDescription
from src.plotting.plotly_styles import TikrStyle
from ..util import proj_types as tp

logger = logging.getLogger("main.strategy_plot")

template = "presentation"  # 'plotly_dark'
font_family = "Raleway"  # "Gravitas One"
font_size = 10
fig_color = "antiquewhite"
bg_color = "blanchedalmond"
buy_color = "chartreuse"
sell_color = "red"
line_color = "thistle"
line_width = 1
grid_color = "rgba(200, 200, 200, 0.2)"
marker_size = 10
marker_opacity = 0.8


# ====================================================================================
class PlotDefinition(NamedTuple):
    name: str
    main: Optional[PlotDescription] = None
    sub: Optional[Sequence[PlotDescription]] = None
    style: TikrStyle = None

    @property
    def number_of_subplots(self) -> int:
        """Calculate the number of subplots needed for the plot."""
        return (
            (1 if self.main is not None else 0)
            + (len(self.sub) if self.sub is not None else 0)
        )

    @property
    def number_of_rows(self) -> int:
        """Calculate the number of rows needed for the plot."""
        return self.number_of_subplots + 1


@dataclass
class LineDefinition:
    label: str
    color: str | None = None
    width: float = 1
    opacity: float = 1
    shape: str = "linear"

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "width": self.width,
            "shape": self.shape,
        }


class TriggerDefinition(LineDefinition):
    #
    #  super().__init__()
    value: float

    def as_dict(self) -> dict:
        return super().as_dict()


@dataclass
class ChannelDefinition:
    label: str
    upper: LineDefinition | str | float | None = None
    lower: LineDefinition | str | float | None = None
    color: str | None = None
    fillmethod: str = "tozeroy"
    opacity: float = 1

    def __post_init__(self):
        for line in [self.upper, self.lower]:
            match line:
                case str():
                    line = LineDefinition(line)
                case LineDefinition() | None:
                    pass
                case _:
                    raise ValueError(f"Invalid line definition: {line}")


# ================================= Plot Functions ===================================
def _prepare_data(data: pd.DataFrame | tp.Data) -> pd.DataFrame:
    return pd.DataFrame.from_dict(data) if isinstance(data, dict) else data


def prepare_fig(p_def: PlotDefinition):
    """Prepare a figure for plotting.

    Parameters
    ----------
    p_def : PlotDefinition
        plot parameters for a strategy
    """
    rows = p_def.number_of_rows
    row_heights = [4] + [3 for _ in range(rows - 1)]
    subplot_titles = [p_def.name] + [sub.label for sub in p_def.sub]

    # secondary y axis for main plot
    specs = [[{"secondary_y": True}]] \
        + [[{"secondary_y": False}] for _ in range(rows-1)]

    if p_def.number_of_subplots == 0:
        fig = make_subplots(
            rows=rows,
            cols=1,
            specs=[specs],
            subplot_titles=subplot_titles
        )
    else:
        fig = make_subplots(
            rows=rows,
            cols=1,
            row_heights=row_heights,
            vertical_spacing=0.05,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            specs=specs,
        )

    fig = add_gridlines(fig, p_def)

    style = p_def.style

    # set some basic style parameters and make the y axis logarithmic
    fig.update_layout(template=template, yaxis_type="log", hovermode="x")

    # set style parameters as defined in the provided style object
    if style is not None:
        fig.update_layout(
            plot_bgcolor=style.colors.background.rgba,
            paper_bgcolor=style.colors.canvas.rgba,
            font_family=style.font_family,
            font_color=style.colors.text.rgba,
            font_size=style.font_size,
            # title_font_family=style.font_family,
            # title_font_color=style.title_font_color,
            # legend_title_font_color=style.legend_title_font_color,
            # showlegend=style.show_legend,
        )
    else:
        fig.update_layout(plot_bgcolor=fig_color, paper_bgcolor=bg_color)

    # set tick font family, size and color
    # fig.update_yaxes(
    #     tickfont=dict(
    #         family=font_family,
    #         color="grey",
    #         size=10,
    #     )
    # )

    # disable range sliders
    fig.update_xaxes(row=1, col=1, rangeslider_visible=False)
    fig.update_xaxes(row=2, col=1, rangeslider_visible=False)
    fig.update_xaxes(rangeslider={"visible": False}, row=p_def.number_of_rows, col=1)
    fig.update_xaxes(row=p_def.number_of_rows, col=1, rangeslider_thickness=0.02)

    fig.update_layout(
        font_family=style.font_family,
        font_color=style.colors.text.rgba,
        font_size=style.font_size,
        title_font_family=font_family,
        title_font_color=style.colors.text.rgba,
        legend_title_font_color=style.colors.text.rgba,
        showlegend=False,
    )

    return fig


def add_gridlines(fig: go.Figure, p_def: PlotDefinition):
    # add_gridlines(fig, color=p_def.style.colors.grid.rgba)
    # make sure the grid is drwan first for all subplots
    # Update grid for all subplots
    rows = p_def.number_of_rows
    color = p_def.style.colors.grid.rgba
    update_params = dict(showgrid=True, gridwidth=0.5, gridcolor=color)
    no_grid_params = dict(showgrid=False)

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


def draw_ohlcv(fig: go.Figure, data: pd.DataFrame, name: str = "no name"):
    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                showlegend=False,
                marker={"color": "rgba(128,128,128,0.15)"},
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
                name=name,
                line=dict(width=0.5),
                showlegend=False,
            ),
            secondary_y=False,
        )

    return fig


def _draw_signal(fig, data: pd.DataFrame, signal_column: str = "signal"):
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
                color=buy_color,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(color=buy_color, width=1),
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
                color=sell_color,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(color=sell_color, width=1),
            ),
            showlegend=False,
        )
    )

    return fig


def draw_buys_and_sells(fig, data: pd.DataFrame):
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=data.index,
            y=data.buy_at,
            marker=dict(
                symbol="triangle-up",
                color=buy_color,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(color=buy_color, width=1),
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
                color=sell_color,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(color=sell_color, width=1),
            ),
            showlegend=False,
        )
    )

    return fig


def draw_positions(fig: go.Figure, data: pd.DataFrame, row=1, col=1):
    # Ensure the dataframe is sorted by index
    data = data.sort_index()

    # Get the 'position' column
    positions = data['position']

    # Identify segments where 'position' remains the same
    segments = positions.ne(positions.shift()).cumsum()

    # Group the dataframe by these segments
    grouped = data.groupby(segments)

    # Define colors for long and short positions
    bull_color = 'rgba(0, 125, 0, 0.075)'  # Green with low opacity
    bear_color = 'rgba(125, 0, 0, 0.075)'  # Red with low opacity

    # Iterate over each group to draw rectangles
    for _, group in grouped:
        pos_value = group['position'].iloc[0]
        if pos_value != 0:
            fig.add_shape(
                type="rect",
                x0=group.index[0],
                x1=group.index[-1],
                y0=0,  # Use 0 for bottom of subplot
                y1=1,  # Use 1 for top of subplot
                xref=f'x{row}' if row > 1 else 'x',
                yref=f'y{row} domain' if row > 1 else 'y domain',
                fillcolor=bull_color if pos_value == 1 else bear_color,
                layer="below",
                line_width=0,
                row=row,
                col=col
            )

    return fig


def draw_main_data(fig, data: pd.DataFrame, p_def: PlotDefinition):
    if p_def.main:
        draw_indicator(fig, data, p_def.main, 0)

    for row in range(1, p_def.number_of_rows + 1):
        draw_positions(fig, data, row=row)

    draw_ohlcv(fig, data)
    draw_buys_and_sells(fig, data)
    # _draw_signal(fig, data)


def draw_indicator(fig, data: pd.DataFrame, desc: PlotDescription, row: int):
    logger.debug(f"drawing indicator {desc.label} in row {row}")

    default_line_def = LineDefinition(
        label=desc.label,
        color=line_color,
        width=0,
        opacity=0,
        shape="spline",
    )

    for channel in desc.channel:

        # convert tuples to ChannelDefinition objects
        if isinstance(desc.channel, tuple):
            desc.channel = ChannelDefinition(
                label=f"{desc.channel[0]} - {desc.channel[1]}",
                upper=default_line_def,
                lower=default_line_def,
            )

        logger.debug("drawing channel: {0}".format(channel))

        if channel.upper:
            # draw upper line
            fig.append_trace(
                go.Scatter(
                    x=data.index,
                    y=data[channel.upper.label],
                    visible=True,
                    name=channel.upper.label,
                    line=channel.upper.as_dict(),
                    opacity=channel.upper.opacity,
                ),
                row=row + 1,
                col=1,
            )

        if channel.lower:
            # draw lower line & fill area
            fig.append_trace(
                go.Scatter(
                    x=data.index,
                    y=data[channel.lower.label],
                    visible=True,
                    name=channel.lower.label,
                    fill=channel.fillmethod,
                    fillcolor=channel.color,
                    line=channel.lower.as_dict(),
                    opacity=channel.lower.opacity,
                ),
                row=row + 1,
                col=1,
            )

    for line in desc.lines:
        logger.debug("drawing line: {0} in row {1}".format(line, row))

        if isinstance(line, tuple):
            line = LineDefinition(line[0])

        fig.append_trace(
            go.Scatter(
                mode="lines",
                x=data.index,
                y=data[line.label],
                name=line.label,
                line=line.as_dict(),
            ),
            row=row + 1,
            col=1,
        )

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

    return fig


def build_figure(data: pd.DataFrame | tp.Data, p_def: PlotDefinition):
    data = _prepare_data(data)
    fig = prepare_fig(p_def)

    # draw main data (candlesticks, buy/sell signals, etc.)
    draw_main_data(fig, data, p_def)

    # draw subplots (if any)
    if p_def.number_of_subplots:
        for row, sub in enumerate(p_def.sub, 1):
            logger.debug("drawing subplot: %s", sub)
            fig = draw_indicator(fig, data, sub, row)

    return fig


def plot(data: pd.DataFrame | tp.Data, p_def: PlotDefinition):
    """
    Plot a strategy.

    Parameters
    ----------
    data: pd.DataFrame | tp.Data
        OHLCV data to plot, possibly with indicator and signal data

    p_def : PlotDefinition
        plot parameters for a strategy
    """
    build_figure(data=data, p_def=p_def).show()


__all__ = ["PlotDefinition", "plot"]
