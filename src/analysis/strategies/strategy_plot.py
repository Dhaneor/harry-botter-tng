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


def _prepare_fig(p_def: PlotDefinition, no_of_subplots: int):
    """Prepare a figure for plotting.

    Parameters
    ----------
    p_def : PlotDefinition
        plot parameters for a strategy
    """
    rows, row_heights, subplot_titles = no_of_subplots + 1, [6], [p_def.name]

    [subplot_titles.append(sub.label) for sub in p_def.sub]
    [row_heights.append(3) for _ in range(rows - 1)]
    main_specs = [{"secondary_y": True}]

    row_heights[-2] = 2

    if no_of_subplots == 0:
        fig = make_subplots(
            rows=rows, cols=1, specs=[main_specs], subplot_titles=subplot_titles
        )

    else:
        specs = [[{"secondary_y": False}]] * (rows - 1)
        specs.insert(0, main_specs)

        fig = make_subplots(
            rows=rows,
            row_heights=row_heights,
            vertical_spacing=0.05,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            specs=specs,
        )

    add_gridlines(fig)

    return fig


def add_gridlines(fig):
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor=grid_color),
    )


def _draw_ohlcv(fig, data: pd.DataFrame, name: str = "uh! oh! we have no name!"):
    candles = go.Candlestick(
        x=data.index,
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close,
        name=name,
        opacity=0.8,
        showlegend=False,
    )

    volume = go.Bar(
        x=data.index,
        y=data["volume"],
        showlegend=False,
        marker={
            "color": "rgba(128,128,128,0.2)",
        },
    )

    fig.add_trace(volume, secondary_y=True)
    fig.add_trace(candles, secondary_y=False)

    return fig


def _draw_main_data(fig, data: pd.DataFrame, desc: PlotDescription):
    _draw_indicator(fig, data, desc, 0)


def _draw_indicator(fig, data: pd.DataFrame, desc: PlotDescription, row: int):
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


def _update_layout(fig, rows: int, style: TikrStyle = None) -> go.Figure:
    # fig.update_traces(opacity=0.7, selector=dict(type="scatter"))
    # fig.update_traces(line_width=0.5, selector=dict(type="scatter"))
    fig.update_traces(line_width=0.5, selector=dict(type="candlestick"))

    # maybe we can use this to color the candlesticks?
    # fig.update_traces(decreasing_line_color="red", selector=dict(type="candlestick"))
    fig.update_layout(template=template, yaxis_type="log", hovermode="x")

    if style is not None:
        logger.info(f"Applying style: {style}")
        fig.update_layout(
            plot_bgcolor=style.colors.background.rgba,
            paper_bgcolor=style.colors.canvas.rgba,
            # font_family=style.font_family,
            # font_color=style.font_color,
            font_size=style.font_size,
            # title_font_family=style.font_family,
            # title_font_color=style.title_font_color,
            # legend_title_font_color=style.legend_title_font_color,
            # showlegend=style.show_legend,
        )
    else:
        fig.update_layout(plot_bgcolor=fig_color, paper_bgcolor=bg_color)

    fig.update_yaxes(
        tickfont=dict(
            family=font_family,
            color="grey",
            size=10,
        )
    )
    fig.update_xaxes(row=1, col=1, rangeslider_visible=False)
    fig.update_xaxes(row=2, col=1, rangeslider_visible=False)
    fig.update_xaxes(rangeslider={"visible": False}, row=rows, col=1)
    fig.update_xaxes(row=rows, col=1, rangeslider_thickness=0.02)

    fig.update_layout(
        font_family=font_family,
        font_color="grey",
        font_size=font_size,
        title_font_family=font_family,
        title_font_color="red",
        legend_title_font_color="green",
        showlegend=True,
    )

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
    data = _prepare_data(data)

    no_of_subplots = len(p_def.sub) if p_def.sub is not None else 0

    fig = _prepare_fig(p_def, no_of_subplots)

    if p_def.main:
        _draw_main_data(fig, data, p_def.main)

    _draw_ohlcv(fig, data)
    _draw_signal(fig, data)

    if no_of_subplots:

        for row, sub in enumerate(p_def.sub, 1):
            logger.debug("drawing subplot: %s", sub)
            fig = _draw_indicator(fig, data, sub, row)

    _update_layout(fig, no_of_subplots + 1, p_def.style)

    fig.show()


__all__ = ["PlotDefinition", "plot"]
