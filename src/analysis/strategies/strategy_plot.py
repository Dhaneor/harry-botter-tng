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
import itertools
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import NamedTuple, Sequence, Optional

from ..indicators.indicator import PlotDescription
from ..util import proj_types as tp

logger = logging.getLogger("main.strategy_plot")
logger.setLevel(logging.DEBUG)

template = "presentation"  # 'plotly_dark'
font_family = "Raleway"  # "Gravitas One"
bg_color = "antiquewhite"
buy_color = "chartreuse"
sell_color = "red"
line_color = "thistle"
marker_size = 10
marker_opacity = 0.8


# ======================================================================================
class PlotDefinition(NamedTuple):
    name: str
    main: Optional[PlotDescription] = None
    sub: Optional[Sequence[PlotDescription]] = None


def _prepare_data(data: pd.DataFrame | tp.Data) -> pd.DataFrame:
    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)

    return data


def _prepare_fig(p_def: PlotDefinition, no_of_subplots: int):
    """Prepare a figure for plotting.

    Parameters
    ----------
    p_def : PlotDefinition
        plot parameters for a strategy
    """
    rows, row_heights, subplot_titles = no_of_subplots + 1, [8], [p_def.name]

    [subplot_titles.append(sub.label) for sub in p_def.sub]
    [row_heights.append(2) for _ in range(rows - 1)]
    main_specs = [{"secondary_y": True}]

    if no_of_subplots == 0:
        fig = make_subplots(
            rows=rows,
            cols=1,
            specs=[main_specs],
            subplot_titles=subplot_titles
        )

    else:
        specs = [[{"secondary_y": False}]] * (rows - 1)
        specs.insert(0, main_specs)

        print(specs)

        fig = make_subplots(
            rows=rows,
            row_heights=row_heights,
            vertical_spacing=0.05,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            specs=specs,
        )

    return fig


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

    # draw channel, if there is one
    for upper, lower in itertools.pairwise(desc.channel):
        chan_line_def = {
            "shape": "linear",
            "color": line_color,
            "width": 0.1,
        }

        logger.debug("drawing channel: {0}".format(upper))
        logger.debug("drawing channel: {0}".format(lower))

        fig.append_trace(
            go.Scatter(
                x=data.index,  # ["human open time"],
                y=data[upper],
                visible=True,
                name=upper,
                line=chan_line_def,
                opacity=0.3,
            ),
            row=row + 1,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=data.index,  # ["human open time"],
                y=data[lower],
                visible=True,
                name=lower,
                fill="tonexty",
                line=chan_line_def,
                opacity=0.3,
            ),
            row=row + 1,
            col=1,
        )

    for line in desc.lines:
        logger.debug("drawing line: {0} in row {1}".format(line, row))

        if row == 0:
            line_def = dict(width=1)
        else:
            line_def = dict(width=1)
            # line_def = dict(color='red', width=1)

        fig.append_trace(
            go.Scatter(
                mode="lines",
                x=data.index,
                y=data[line[0]],
                name=desc.label,
                line=line_def,
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


def _update_layout(fig, rows: int):
    fig.update_traces(opacity=0.7, selector=dict(type="scatter"))
    fig.update_traces(line_width=0.5, selector=dict(type="scatter"))
    fig.update_traces(line_width=0.5, selector=dict(type="candlestick"))

    # maybe we can use this to color the candlesticks?
    # fig.update_traces(decreasing_line_color="red", selector=dict(type="candlestick"))

    fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    fig.update_layout(template=template, yaxis_type="log", hovermode="x")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(
        tickfont=dict(
            family=font_family,
            color="grey",
            size=10,
        )
    )
    fig.update_xaxes(row=1, col=1, rangeslider_visible=False)
    fig.update_xaxes(row=2, col=1, rangeslider_visible=False)
    fig.update_xaxes(rangeslider={"visible": True}, row=rows, col=1)
    fig.update_xaxes(row=rows, col=1, rangeslider_thickness=0.02)

    fig.update_layout(
        font_family=font_family,
        font_color="grey",
        font_size=10,
        title_font_family=font_family,
        title_font_color="red",
        legend_title_font_color="green",
        showlegend=False,
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
            fig = _draw_indicator(fig, data, sub, row)

    _update_layout(fig, no_of_subplots + 1)

    fig.show()


__all__ = ["PlotDefinition", "plot"]
