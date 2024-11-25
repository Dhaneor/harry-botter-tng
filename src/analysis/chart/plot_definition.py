#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
import pandas as pd
import plotly.graph_objects as go

from typing import Sequence, NamedTuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Color:
    def __init__(self, red=0, green=0, blue=0, alpha=1.0):
        self.r: int = red
        self.g: int = green
        self.b: int = blue
        self._a: float = alpha

        self.initial: tuple = (self.r, self.g, self.b, self.a)

    # def __str__(self) -> str:
    #     return self.rgba

    def __repr__(self) -> str:
        return f"Color(r={self.r}, g={self.g}, b={self.b}, a={self.a})"

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, alpha: float) -> None:
        if alpha is None:
            return

        if not 0 <= alpha <= 1.0:
            raise ValueError("Alpha value must be between 0.0 and 1.0")

        self._a = alpha
        self.initial = (self.r, self.g, self.b, self._a)

    def set_alpha(self, alpha: float) -> "Color":
        if alpha is None:
            return self

        if not 0 <= alpha <= 1.0:
            raise ValueError("Alpha value must be between 0.0 and 1.0")

        # self.a = alpha
        return Color(self.r, self.g, self.b, alpha)

    def reset(self) -> "Color":
        self.r, self.g, self.b, self.a = self.initial
        return self

    # ................................................................................
    @property
    def hex(self) -> str:
        return f"#{hex(self.r)[2:]}{hex(self.g)[2:]}{hex(self.b)[2:]}"

    @property
    def rgb(self) -> str:
        return f"rgb({self.r}, {self.g}, {self.b})"

    @property
    def rgba(self) -> str:
        r, g, b, a = self.rgba_tuple
        return f"rgba({r}, {g}, {b}, {a})"

    @property
    def rgba_tuple(self) -> tuple[int, int, int, float]:
        return (self.r, self.g, self.b, self.a)

    # ................................................................................
    @classmethod
    def from_hex(cls, hex_color: str, alpha: float = 1.0) -> "Color":

        def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

        color = cls(*(*hex_to_rgb(hex_color), alpha))
        color.a = alpha
        return color

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, alpha: float = 1.0) -> "Color":
        return cls(r, g, b, alpha)


@dataclass
class Colors:
    strategy: Color
    capital: Color
    hodl: Color
    candle_up: Color
    candle_down: Color
    buy: Color
    sell: Color
    volume: Color

    canvas: Color
    background: Color
    grid: Color
    text: Color

    # fill colors are set automatically by the __post_init__ method
    strategy_fill: Color = None
    capital_fill: Color = None
    hodl_fill: Color = None

    def __post_init__(self):
        # turn the colors into Color objects
        self.strategy = Color.from_hex(self.strategy)
        self.capital = Color.from_hex(self.capital)
        self.hodl = Color.from_hex(self.hodl)
        self.candle_up = Color.from_hex(self.candle_up)
        self.candle_down = Color.from_hex(self.candle_down)
        self.buy = Color.from_hex(self.buy)
        self.sell = Color.from_hex(self.sell)
        self.volume = Color.from_hex(self.volume)

        self.canvas = Color.from_hex(self.canvas)
        self.background = Color.from_hex(self.background)
        self.grid = Color.from_hex(self.grid)
        self.text = Color.from_hex(self.text)

    def add_fill_colors(self, fill_alpha) -> None:
        self.strategy_fill = Color(*self.strategy.rgba_tuple).set_alpha(fill_alpha)
        self.capital_fill = Color(*self.capital.rgba_tuple).set_alpha(fill_alpha)
        self.hodl_fill = Color(*self.hodl.rgba_tuple).set_alpha(fill_alpha)

    def update_line_alpha(self, alpha: float) -> None:
        self.strategy.alpha = alpha
        self.capital.alpha = alpha
        self.hodl.alpha = alpha
        self.candle_up.alpha = alpha
        self.candle_down.alpha = alpha
        self.buy.alpha = alpha
        self.sell.alpha = alpha

    # ................................................................................
    @classmethod
    def from_palette(self, palette: Sequence[str]) -> "Colors":
        return Colors(
            strategy=palette[0],
            capital=palette[1],
            hodl=palette[2],
            candle_up=palette[3],
            candle_down=palette[4],
            buy=palette[5],
            sell=palette[6],
            volume=palette[7],
            canvas=palette[8],
            background=palette[9],
            grid=palette[10],
            text=palette[11],
        )


@dataclass
class TikrStyle:
    colors: Colors
    line_width: float = 1

    line_alpha: float = 0.75
    fill_alpha: float = 0.2
    shadow_alpha: float = 0.2

    candle_up_alpha: float = 1
    candle_down_alpha: float = 1

    font_family: str = "Arial"
    font_size: int = 12
    tick_font_size: int = 10

    volume_opacity: float = 1

    marker_size: int = 10
    marker_opacity: float = 1

    canvas_image: str = None
    canvas_image_opacity: float = 1.0

    def __post_init__(self):
        self.colors.add_fill_colors(self.fill_alpha)
        self.colors.update_line_alpha(self.line_alpha)


@dataclass
class Line:
    label: str | None = None
    column: str | None = None
    color: Color | None = None
    width: float = 1
    opacity: float = 1
    shape: str = "spline"
    zorder: int | None = 0
    legendgroup: str | None = None

    def add_trace(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        row: int = 1,
        col: int = 1,
        fill: str | None = None,
        fillcolor: str | None = None,
    ) -> go.Figure:

        logger.debug(f"Adding line '{self.label}' to plot.")

        fillcolor = fillcolor.rgba if fillcolor is not None else self.color.rgba

        self._add_line_shadow(fig, data, row, col)

        trace = go.Scatter(
            x=data.index,
            y=data[self.column or self.label],
            name=self.label or self.column,
            line=self.as_dict(),
            opacity=self.opacity,
            fill=fill,
            fillcolor=fillcolor,
            showlegend=True if self.label is not None else False,
            legendgroup=self.legendgroup,
            zorder=self.zorder,
        )
        fig.add_trace(trace, row=row, col=col)

        return fig

    def _add_line_shadow(self, fig, data, row=1, col=1) -> go.Figure:
        # shadow_color = self.color.rgba.replace(f"{self.color.a}", "0.1")
        for factor in (3, 2):
            shadow_color = Color(
                self.color.r / factor,
                self.color.g / factor,
                self.color.b / factor,
                self.color.a / factor).rgba  # replace with lighter color

            # shadow_data = data.copy()
            # shadow_data[self.label] = data[self.label]  # * 0.9

            shadow_trace = go.Scatter(
                x=data.index,
                y=data[self.column or self.label],
                name=f"Shadow_{self.label}",
                line=dict(
                    color=shadow_color, width=self.width + factor, shape=self.shape
                    ),
                opacity=self.opacity,
                fill=None,
                fillcolor=None,
                hoverinfo='skip',
                showlegend=False,
                zorder=self.zorder - 1,
            )
            fig.add_trace(shadow_trace, row=row, col=col)

        return fig

    def as_dict(self) -> dict:
        return {
            "color": self.color.rgba,
            "width": self.width,
            "shape": self.shape,
        }


class TriggerDefinition(Line):
    value: float

    def as_dict(self) -> dict:
        return super().as_dict()


@dataclass
class Channel:
    label: str
    upper: Line | str | float | None = None
    lower: Line | str | float | None = None
    color: Color | None = None
    fillmethod: str = "tonexty"
    opacity: float = 1

    def __post_init__(self):
        for line in [self.upper, self.lower]:
            match line:
                case str():
                    line = Line(line)
                case Line() | None:
                    pass
                case _:
                    raise ValueError(f"Invalid line definition: {line}")

    def add_trace(self, fig, data, row=1, col=1, fill_alpha=None) -> go.Figure:
        if self.upper is not None:
            self.upper.add_trace(fig, data, row=row, col=col)

        # fill_alpha = fill_alpha if fill_alpha is not None else self.color.a

        if self.lower is not None:
            self.lower.add_trace(
                fig,
                data,
                row=row,
                col=col,
                fill=self.fillmethod,
                fillcolor=self.color.set_alpha(fill_alpha),
            )

        return fig


@dataclass
class Drawdown:
    label: str
    column: str
    legendgroup: str | None = None
    line: Line | str | float | None = None  # line for the dd levels
    color: Color | None = None
    gradient: bool = True,
    critical: float | None = None
    opacity: float = 1
    color_scale: list[tuple[float, str]] | None = None

    def __post_init__(self):
        for line in [self.line]:
            match line:
                case str():
                    line = Line(line)
                case Line() | None:
                    pass
                case _:
                    raise ValueError(f"Invalid line definition: {line}")

        if not self.color_scale:
            self.color_scale = [
                (0, self.color.set_alpha(self.line.color.a).rgba),
                (0.1, self.color.set_alpha(self.line.color.a).rgba),
                (0.8, self.color.reset().rgba),
                (1, self.color.reset().rgba),
            ]

        self.line.label = self.label
        self.line.column = self.column
        self.line.legendgroup = self.legendgroup if self.legendgroup is not None \
            else self.line.legendgroup

    def add_trace(self, fig, data, row=1, col=1, fill_alpha=None) -> go.Figure:
        if self.gradient:

            dd = go.Scatter(
                x=data.index,
                y=data[self.column],
                name=self.label,
                fill="tozeroy",
                line=self.line.as_dict(),
                fillgradient=dict(type="vertical", colorscale=self.color_scale),
                showlegend=True if self.label is not None else False,
                legendgroup=self.legendgroup if self.legendgroup is not None else None,
                opacity=self.opacity,
            )
            fig.add_trace(dd, row=row, col=col)

        else:
            fill_alpha = fill_alpha if fill_alpha is not None else self.color.a

            self.line.add_trace(
                    fig,
                    data,
                    row=row,
                    col=col,
                    fill="tozeroy",
                    fillcolor=self.color.set_alpha(fill_alpha)
                )

        if self.critical is not None:
            logger.info(f"Critical drawdown level: {self.critical}")
            logger.info(f"Minimum drawdown level: {data[self.column].min()}")

            try:
                critical_level = round(
                    1 - ((self.critical - 5) / (data[self.column].min())), 4
                )

                threshhold = round(
                    1 - ((self.critical + 2) / (data[self.column].min())), 4
                )

            except ZeroDivisionError:
                # Critical level cannot be calculated (= there was no
                # drawdown at all), so do not add a 'critical' fill area
                return fig

            overshoot = abs(critical_level) if critical_level < 0 else 0

            if critical_level < 0 and threshhold < 0:
                logger.info("Critical level is below the max drawdown level")
                return fig

            if critical_level <= 0 and threshhold > 0:
                critical_level = 0.001
                overshoot = critical_level - 1

            logger.info(
                f"Critical level: {critical_level} | Threshhold: {threshhold}"
                )

            critical = go.Scatter(
                x=data.index,
                y=data[self.column],
                fill="tozeroy",
                line=dict(width=0),  # self.line.as_dict(),
                fillgradient=dict(
                    type="vertical",
                    colorscale=[
                        (0, f"rgba(255, 0, 0, {0.5 - overshoot})"),
                        (critical_level, "rgba(255, 0, 0, 0.5)"),
                        (threshhold, self.color.set_alpha(0.5).rgba),
                        (1, self.color.set_alpha(0).rgba),
                    ],
                ),
                showlegend=False,
                hoverinfo='skip'
            )

            fig.add_trace(critical, row=row, col=col)

        return fig


# ====================================================================================
@dataclass(frozen=True)
class PlotDescription:
    """Plot description for one indicator.

    This description is used to tell the chart plotting component how
    to plot the indicator. Every indicator builds and returns this
    automatically

    Attributes
    ----------
    label: str
        the label for the indicator plot

    is_subplot: bool
        does this indicator require a subplot or is it layered with
        the OHLCV data in the main plot

    lines: Sequence[tuple[str, str]]
        name(s) of the indicator values that were returned with the
        data dict, or are then a column name in the dataframe that
        is sent to the chart plotting component

    triggers: Sequence[tuple[str, str]]
        name(s) of the trigger values, these are separated from the
        indicator values to enable a different reprentation when
        plotting

    channel: tuple[str, str]
        some indicators require plotting a channel, this makes it
        clear to the plotting component, which data series to plot
        like this.

    level: str
        the level that produced this description, just to make
        debugging easier
    """

    label: str
    is_subplot: bool
    lines: Sequence[tuple[str, str]] = field(default_factory=list)
    triggers: Sequence[tuple[str, str]] = field(default_factory=list)
    channel: list[str] = field(default_factory=list)
    hist: list[str] = field(default_factory=list)
    level: str = field(default="indicator")


class PlotDefinition(NamedTuple):
    name: str
    main: PlotDescription | None = None
    sub: PlotDescription | None = None
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
