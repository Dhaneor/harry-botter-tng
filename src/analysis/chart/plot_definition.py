#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import itertools
import logging
import pandas as pd
import plotly.graph_objects as go
import sys

from abc import ABC, abstractmethod
from contextlib import contextmanager
from random import random, choice
from typing import Sequence, List, Dict, Tuple, Literal, Iterator
from dataclasses import dataclass, field

if __name__ == "__main__":
    from layout_validator import validate_layout

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a StreamHandler to output log messages to the console
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
else:
    from .layout_validator import validate_layout

    logger = logging.getLogger(f"main.{__name__}")
    logger.setLevel(logging.ERROR)


VALIDATION_LEVEL = "basic"

Specs = Tuple[Tuple[Dict]]


# ========================== Classes for style definitions ===========================
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
    font_opacity: float = 1
    font_size: int = 12
    title_font_size: int = 16
    tick_font_size: int = 10

    volume_opacity: float = 1

    marker_size: int = 10
    marker_opacity: float = 1

    canvas_image: str = None
    canvas_image_opacity: float = 1.0

    def __post_init__(self):
        self.colors.add_fill_colors(self.fill_alpha)
        self.colors.update_line_alpha(self.line_alpha)


# ======================= Classes for basic chart components =========================
@dataclass
class ChartElement(ABC):
    label: str | None = None
    column: str | None = None
    legendgroup: str | None = None
    legend: str | None = None

    font: str | None = None
    font_size: int | None = None

    row: int = 1
    col: int = 1

    width: float | None = None
    color: Color | None = None
    opacity: float = 1
    zorder: int | None = 0
    secondary_y: bool = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(label={self.label}, column={self.column}, "
            f"legendgroup={self.legendgroup}, legend={self.legend}, row={self.row}, "
            f"col={self.col}"
        )

    @abstractmethod
    def apply_style(self, style: TikrStyle) -> None: ...

    @abstractmethod
    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure: ...

    def adjust_and_convert_colorscale(
        self, color_scale, max_leverage
    ) -> list[list[float, Color]]:
        """
        Adjusts a colorscale based on the maximum leverage value
        and converts it to Plotly's relative format.

        Parameters:
        -----------
        color_scale: List[Tuple[float, str]],
            List of tuples [(value, color), ...] with absolute values.
        max_leverage: float
            Maximum leverage value.

        Returns:
        plotly_colorscale:
            List of [relative_position, color] suitable for Plotly.
        """
        # Sort the color scale by value
        color_scale = sorted(color_scale, key=lambda x: x[0])

        # Remove color stops beyond max_leverage and interpolate if necessary
        adjusted_scale = []
        for i, (value, color) in enumerate(color_scale):
            if value < max_leverage:
                adjusted_scale.append((value, color))
            elif value == max_leverage:
                adjusted_scale.append((value, color))
                break
            else:
                # Interpolate color at max_leverage
                if i == 0:
                    # Max leverage is less than the first value; use the first color
                    adjusted_scale.append((max_leverage, color))
                else:
                    prev_value, prev_color = color_scale[i - 1]
                    fraction = (max_leverage - prev_value) / (value - prev_value)
                    color_interp = self.interpolate_color(
                        prev_color, color, fraction
                        )
                    adjusted_scale.append((max_leverage, color_interp))
                break

        # Normalize the adjusted colorscale values to [0, 1]
        min_value = adjusted_scale[0][0]
        max_value = adjusted_scale[-1][0]
        value_range = max_value - min_value if max_value != min_value else 1

        plotly_colorscale = []
        for value, color in adjusted_scale:
            relative_position = (value - min_value) / value_range
            plotly_colorscale.append([relative_position, color])

        return plotly_colorscale

    def interpolate_color(self, color_1: Color, color_2: Color, fraction: float):
        """
        Interpolates between two colors by a given fraction.

        Parameters:
        - color_1: Starting color string.
        - color_2: Ending color string.
        - fraction: Fraction between 0 and 1 representing the interpolation amount.

        Returns:
        - color_str: Interpolated color string in 'rgba(r, g, b, a)' format.
        """
        color1 = color_1.rgba_tuple
        color2 = color_2.rgba_tuple

        r = color1[0] + (color2[0] - color1[0]) * fraction
        g = color1[1] + (color2[1] - color1[1]) * fraction
        b = color1[2] + (color2[2] - color1[2]) * fraction
        a = color1[3] + (color2[3] - color1[3]) * fraction

        return Color(r, g, b, a)


@dataclass
class Candlestick(ChartElement):
    color_up: Color | None = None
    color_down: Color | None = None
    candle_up_alpha: float = 1
    candle_down_alpha: float = 1

    def apply_style(self, style: TikrStyle) -> None:
        self.color_up = style.colors.candle_up
        self.color_down = style.colors.candle_down

        self.candle_up_alpha = style.candle_up_alpha
        self.candle_down_alpha = style.candle_down_alpha

        self.opacity = self.candle_up_alpha

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        logger.debug(f"Adding candlestick '{self.label}' to plot.")

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                name=self.label or "Candlestick",
                line=dict(width=0.75),
                opacity=self.opacity,
                showlegend=False,
                legendgroup=self.label,
                zorder=self.zorder,
            ),
            row=self.row,
            col=self.col,
        )

        return fig


@dataclass
class Signal(ChartElement):
    """Base class for signals.

    Subclasses should implement the apply_style method.

    Arguments:
    ----------
    Inherited from ChartElement:
        label: str, optional
            The label for the signal that is is used for the legend
        column: str
            The column that contains the signal data (=y position)
        color: Color, optional
            A Color instance representing the color of the signal.
            NOTE: The color will be set by the container when applying
            the style if no color is set durin ginstantiation of the
            class.
        opacity: float, optional
            This is intended to increase or decrease the opacity for
            all elements within a SubPlot, controlled by an external
            method. Use the Color and RGBA values to set the color
            and the base transparancy and leave this at the default
            value of 1.0.
        zorder: int, optional
            The z-order determines the order in which traces and shapes
            are drawn. This determines which trace appears on top of
            other traces and shapes. The default is 0.
        secondary_y: bool, optional
            Inherited, but not applicabgle for this class.

    type: Literal['up', 'down', 'neutral']
        Sets the type of signal (up, down, neutral), default 'neutral'
    marker_size: int, optional
        The size of the marker for the signal, default 10
    marker_type: str, optional
        The type of marker for the signal, default 'circle'. See the
        Plotly documentation for more marker types.

    Returns:
    --------
    Signal
        Instance of Signal

    Raises:
    -------
    NotImplementedError:
        When used directly, without implementing the apply_style method.
    """

    type: Literal["up", "down", "neutral"] = "neutral"
    marker_size: int | None = None
    marker_symbol_up: str = "circle-dot"
    marker_symbol_down: str = "circle-dot"
    marker_symbol_neutral: str = "circle-dot"

    def apply_style(self, style: TikrStyle) -> None:
        raise NotImplementedError(
            f"apply_style method not implemented for {self.label} class"
        )

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        logger.debug(f"Adding {self.type}s '{self.label}' to plot.")

        match self.type:
            case "up":
                marker_symbol = self.marker_symbol_up
            case "down":
                marker_symbol = self.marker_symbol_down
            case "neutral":
                marker_symbol = self.marker_symbol_neutral
            case _:
                raise ValueError(f"Invalid signal type: {self.type}")

        marker = go.Scatter(
            x=data.index,
            y=data[self.column or self.label],
            name=self.label or self.column,
            mode="markers",
            marker=dict(color=self.color.rgba, size=self.marker_size),
            opacity=self.opacity,
            legend=self.legend,
            legendgroup=self.legendgroup,
            showlegend=True if self.label is not None else False,
            marker_symbol=marker_symbol,
            zorder=self.zorder,
        )
        fig.add_trace(marker, row=self.row, col=self.col)

        return fig


@dataclass
class Line(ChartElement):
    shape: str = "spline"
    fillmethod: str | None = None
    fillcolor: Color | None = None

    shadow: bool = True
    glow: bool = False
    end_marker: bool = False

    is_trigger: bool = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(label={self.label}, column={self.column}, "
            f"legendgroup={self.legendgroup}, legend={self.legend}, row={self.row}, "
            f"col={self.col}, is_trigger={self.is_trigger})"
        )

    def apply_style(self, style: TikrStyle) -> None:
        self.width = style.line_width if not self.width else self.width
        self.font = style.font_family if not self.font else self.font
        self.font_size = style.tick_font_size if not self.font_size else self.font_size

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:

        logger.debug(
            f"Adding line '{self.label}' with fill "
            f"({self.fillmethod}: {self.fillcolor}) to plot."
        )

        logger.debug(self)

        if self.shadow:
            logger.debug(f"Adding line shadow for '{self.label}'")
            self._add_line_shadow(fig, data)

        trace = go.Scatter(
            x=data.index,
            y=data[self.column or self.label],
            name=self.label or self.column,
            line=self.as_dict(),
            opacity=self.opacity,
            fill=self.fillmethod,
            fillcolor=self.fillcolor.rgba if self.fillcolor else None,
            showlegend=True if self.label is not None else False,
            legendgroup=self.legendgroup,
            zorder=self.zorder,
        )
        fig.add_trace(trace, row=self.row, col=self.col)

        if self.end_marker:
            end_marker = go.Scatter(
                x=[data.index[-1]],
                y=[int(data[self.column or self.label].astype(int).iloc[-1])],
                mode="markers+text",
                marker=dict(color=self.color.rgba, size=5),
                text=[data[self.column or self.label].iloc[-1].astype(int)],
                textfont=dict(
                    family=self.font,
                    size=self.font_size,  # Use the font_size attribute here
                    color=self.color.rgba,
                ),
                textposition="middle right",
                showlegend=False,
            )

            fig.add_trace(end_marker, row=self.row, col=self.col)

        return fig

    def _add_line_shadow(self, fig, data) -> go.Figure:
        for factor in (5, 2.5, 1.5):
            color = Color(
                self.color.r / factor,
                self.color.g / factor,
                self.color.b / factor,
                self.color.a / factor,
            ).rgba  # replace with darker color

            line = dict(
                color=color,
                width=self.width + factor,
                shape=self.shape,
                smoothing=1.3 if self.shape == "spline" else None,
            )

            shadow_trace = go.Scatter(
                x=data.index,
                y=data[self.column or self.label],
                line=line,
                opacity=self.opacity,
                hoverinfo="skip",
                showlegend=False,
                zorder=round(self.zorder - factor),
            )
            fig.add_trace(shadow_trace, row=self.row, col=self.col)

        return fig

    def as_dict(self) -> dict:
        return {
            "color": self.color.rgba,
            "width": self.width,
            "shape": self.shape,
        }


class Trigger(Line):
    value: float
    is_trigger = True

    def as_dict(self) -> dict:
        return super().as_dict()


@dataclass
class Channel:
    label: str
    upper: Line | str | float | None = None
    lower: Line | str | float | None = None

    _color: Color = None
    _shadow: bool = True
    fillmethod: str = "tonexty"

    _legendgroup: str = None
    _row: int = None
    _col: int = None

    def __repr__(self) -> str:
        return f"Channel {self.label}\n\t\tupper: {self.upper}\n\t\tlower: {self.lower}"

    def __post_init__(self):
        self.lower.fillmethod = self.fillmethod
        self.lower.zorder = -5
        self.upper.zorder = -5

    @property
    def legendgroup(self) -> str:
        return self._legendgroup or self.label

    @legendgroup.setter
    def legendgroup(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"Invalid legendgroup: {value}")

        self._legendgroup = value
        self.upper.legendgroup = value
        self.lower.legendgroup = value

    @property
    def row(self) -> int:
        return self._row

    @row.setter
    def row(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"Invalid row: {value}")

        self._row = value
        self.upper.row = value
        self.lower.row = value

    @property
    def col(self) -> int:
        return self._col

    @col.setter
    def col(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"Invalid col: {value}")

        self._col = value
        self.upper.col = value
        self.lower.col = value

    @property
    def color(self) -> Color:
        return self._color
    
    @color.setter
    def color(self, value: Color):
        # self._color = value
        self.upper.color = value
        self.lower.color = value

    @property
    def shadow(self) -> bool:
        return self._shadow
    
    @shadow.setter
    def shadow(self, value: bool):
        self._shadow = value
        self.upper.shadow = value
        self.lower.shadow = value

    def apply_style(self, style: TikrStyle) -> None:
        self.upper.apply_style(style)
        self.lower.apply_style(style)

        self.lower.fillcolor = (
            self.color if self.color else self.lower.color.set_alpha(style.fill_alpha)
        )

    def add_trace(self, fig, data) -> go.Figure:
        if self.upper is not None:
            self.upper.add_trace(fig, data)

        if self.lower is not None:
            self.lower.add_trace(fig, data)

        fig.add_trace(
            go.Scatter(
                x=[data.index[-1]],
                y=data[self.upper.column],
                mode="lines",
                fill=self.fillmethod,
                fillcolor=self.color.rgba,
                line=dict(width=0, color=self.lower.color.rgba),
                hoverinfo="skip",
                legendgroup=self.legendgroup,
                name=self.label,
                showlegend=False,
            ),
            row=self.row,
            col=self.col,
        )

        return fig


class Histogram(ChartElement):
    shape: str = "spline"
    fillmethod: str | None = None
    fillcolor: Color | None = None
    border_color: Color | None = None
    opacity: float = 0.5

    is_trigger: bool = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(label={self.label}, column={self.column}, "
            f"legendgroup={self.legendgroup}, legend={self.legend}, row={self.row}, "
            f"col={self.col}, is_trigger={self.is_trigger})"
        )

    def apply_style(self, style: TikrStyle) -> None:
        self.width = style.line_width if not self.width else self.width
        self.font = style.font_family if not self.font else self.font
        self.font_size = style.tick_font_size if not self.font_size else self.font_size

        self.color = style.colors.volume
        self.border_color = style.colors.volume.set_alpha(self.color.a / 2)

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:

        logger.debug(f"Adding histogram '{self.label}' to plot.")
        logger.debug(self)

        trace = go.Bar(
            x=data.index,
            y=data[self.column or self.label],
            name=self.label or self.column,
                marker={
                    "color": self.color.rgba,
                    "line": {"color": self.border_color.rgba, "width": self.width},
                },
            opacity=self.opacity,
            showlegend=True if self.label is not None else False,
            legendgroup=self.legendgroup,
            zorder=self.zorder,
        )
        fig.add_trace(trace, row=self.row, col=self.col)

        return fig

    def _add_line_shadow(self, fig, data) -> go.Figure:
        for factor in (5, 2.5, 1.5):
            color = Color(
                self.color.r / factor,
                self.color.g / factor,
                self.color.b / factor,
                self.color.a / factor,
            ).rgba  # replace with darker color

            line = dict(
                color=color,
                width=self.width + factor,
                shape=self.shape,
                smoothing=1.3 if self.shape == "spline" else None,
            )

            shadow_trace = go.Scatter(
                x=data.index,
                y=data[self.column or self.label],
                line=line,
                opacity=self.opacity,
                hoverinfo="skip",
                showlegend=False,
                zorder=round(self.zorder - factor),
            )
            fig.add_trace(shadow_trace, row=self.row, col=self.col)

        return fig

    def as_dict(self) -> dict:
        return {
            "color": self.color.rgba,
            "width": self.width,
            "shape": self.shape,
        }



# ========================== Classes for chart components ============================
@dataclass
class Buy(Signal):
    type = "up"
    label: str = "BUY"
    column: str = "buy_at"

    def apply_style(self, style: TikrStyle) -> None:
        self.color = self.color or style.colors.buy.set_alpha(style.marker_opacity)
        self.marker_size = self.marker_size or style.marker_size


@dataclass
class Sell(Signal):
    type = "down"
    label: str = "SELL"
    column: str = "sell_at"

    def apply_style(self, style: TikrStyle) -> None:
        self.color = self.color or style.colors.sell.set_alpha(style.marker_opacity)
        self.marker_size = self.marker_size or style.marker_size


@dataclass
class Sells(Signal):
    type = "down"
    label: str = "SELL"
    column: str = "sell_at"

    def apply_style(self, style: TikrStyle) -> None:
        self.color = style.colors.buy.set_alpha(style.marker_opacity)
        self.marker_size = style.marker_size


@dataclass
class Volume(ChartElement):
    border_color: Color | None = None
    secondary_y: bool = True
    zorder: int = -1

    def apply_style(self, style: TikrStyle) -> None:
        self.color = style.colors.volume.set_alpha(style.volume_opacity)
        self.border_color = style.colors.volume.set_alpha(self.color.a / 2)

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        logger.debug(f"Adding volume '{self.label}' to plot.")

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[self.column or self.label],
                name=self.label or self.column,
                showlegend=False,
                marker={
                    "color": self.color.rgba,
                    "line": {"color": self.border_color.rgba, "width": self.width},
                },
                opacity=self.opacity,
                legendgroup=self.legendgroup,
                zorder=self.zorder,
            ),
            secondary_y=self.secondary_y,
            row=self.row,
            col=self.col,
        )

        return fig


@dataclass
class Positions(ChartElement):
    long_color: Color | None = None
    short_color: Color | None = None
    zorder: int = -5

    def apply_style(self, style: TikrStyle) -> None:
        self.long_color = style.colors.buy.set_alpha(style.fill_alpha)
        self.short_color = style.colors.sell.set_alpha(style.fill_alpha)

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        # Identify segments where 'position' remains the same
        positions = data["position"]
        segments = positions.ne(positions.shift()).cumsum()

        # Iterate over each group to draw rectangles
        for _, group in data.groupby(segments):
            pos_value = group["position"].iloc[0]
            if pos_value != 0:
                fillcolor = self.long_color if pos_value == 1 else self.short_color

                fillcolor = fillcolor.set_alpha(fillcolor.a / 2).rgba

                fig.add_shape(
                    type="rect",
                    x0=group.index[0],
                    x1=group.index[-1],
                    y0=0,  # Use 0 for bottom of subplot
                    y1=1,  # Use 1 for top of subplot
                    xref=f"x{self.row}" if self.row > 1 else "x",
                    yref=f"y{self.row} domain" if self.row > 1 else "y domain",
                    fillcolor=fillcolor,
                    layer="below",
                    line_width=0,
                    row=self.row,
                    col=self.col,
                )

        return fig


@dataclass
class Drawdown(Line):
    gradient: bool = (True,)
    critical: float | None = None
    color_scale: list[tuple[float, str]] | None = None

    def __post_init__(self):
        if not self.color_scale:
            self.color_scale = [
                (0, self.color.set_alpha(self.color.a).rgba),
                (0.1, self.color.set_alpha(self.color.a).rgba),
                (0.8, self.color.reset().rgba),
                (1, self.color.reset().rgba),
            ]

        self.fillmethod = "tozeroy"

    def apply_style(self, style: TikrStyle) -> None:
        super().apply_style(style)

        self.fillcolor = (
            self.color.set_alpha(style.fill_alpha)
            if not self.fillcolor
            else self.fillcolor
        )

    def add_trace(self, fig, data) -> go.Figure:
        if self.shadow:
            self._add_line_shadow(fig, data)

        if self.gradient:
            fillgradient = dict(type="vertical", colorscale=self.color_scale)
            fillcolor = None
        else:
            fillgradient = None
            if self.fillcolor is None:
                raise ValueError(f"Fill color must be defined for {self.label}")
            else:
                fillcolor = self.fillcolor.rgba

        dd = go.Scatter(
            x=data.index,
            y=data[self.column],
            name=self.label,
            fill=self.fillmethod,
            line=self.as_dict(),
            fillcolor=fillcolor,
            fillgradient=fillgradient,
            showlegend=True if self.label is not None else False,
            legendgroup=self.legendgroup,
            opacity=self.opacity,
        )
        fig.add_trace(dd, row=self.row, col=self.col)

        # add critical level fill area, if critical level is defined
        if self.critical is not None:
            logger.debug(f"Critical drawdown level: {self.critical}")
            logger.debug(f"Minimum drawdown level: {data[self.column].min()}")

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
                logger.debug("Critical level is below the max drawdown level")
                return fig

            if critical_level <= 0 and threshhold > 0:
                critical_level = 0.001
                overshoot = critical_level - 1

            logger.debug(f"Critical level: {critical_level} | Threshhold: {threshhold}")

            critical = go.Scatter(
                x=data.index,
                y=data[self.column],
                fill="tozeroy",
                line=dict(width=0),
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
                hoverinfo="skip",
            )

            fig.add_trace(critical, row=self.row, col=self.col)

        return fig


@dataclass
class Leverage(ChartElement):
    fillcolor: Color | None = None
    fillmethod: str = "tozeroy"
    fillstep: float = 0.25
    alphastep: float = 0.1

    no_leverage_area_color: Color | None = None

    zorder: int = 1

    def apply_style(self, style: TikrStyle) -> None:
        self.width = style.line_width if not self.width else self.width
        self.color = self.color if self.color else style.colors.strategy
        self.fillcolor = self.fillcolor \
            if self.fillcolor else style.colors.strategy_fill

        self.no_leverage_area_color = self.no_leverage_area_color \
            if self.no_leverage_area_color \
            else style.colors.buy.set_alpha(style.fill_alpha / 2)

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        # set leverage to zero for times where we had no position
        data.loc[data['position'] == 0, 'leverage'] = 0

        # draw a filled rectangular area that goes across the whole
        # length of te subplot and from 0 to 1, which has the same
        # color as the position rectangles
        fig.add_shape(
            type="rect",
            x0=data.index[0],
            x1=data.index[-1],
            y0=0,
            y1=min(1 / data[self.column].max(), 1),
            xref=f"x{self.row}" if self.row > 1 else "x",
            yref=f"y{self.row} domain" if self.row > 1 else "y domain",
            fillcolor=self.no_leverage_area_color.rgba,
            layer="below",
            line_width=0,
            row=self.row,
            col=self.col,
        )

        # define the colors for the different zones
        safe_zone_color_start = Color(0, 200, 0, 0.6)
        safe_zone_color_stop = Color(0, 200, 0, 0.3)
        less_safe_color_start = Color(255, 255, 0, 0.4)
        less_safe_color_stop = Color(255, 128, 0, 0.6)
        danger_color_start = Color(255, 64, 0, 0.6)
        danger_color_stop = Color(200, 32, 0, 0.8)

        # a color scale that represents the increasing danger
        # of higher leverage
        color_scale = [
            (0, safe_zone_color_start),
            (1, safe_zone_color_stop),
            (1.5, less_safe_color_start),
            (2, less_safe_color_stop),
            (2.5, danger_color_start),
            (3, danger_color_stop),
            (10, danger_color_stop.set_alpha(1)),
        ]

        color_scale = self.adjust_and_convert_colorscale(
            color_scale,
            data['leverage'].max()
        )

        # convert all Color objects to RGBA strings
        color_scale = [(stop, color.rgba) for stop, color in color_scale]

        # draw the fill area without a surrounding line
        leverage = go.Scatter(
            x=data.index,
            y=data[self.column],
            name=self.label,
            mode="lines",
            line=dict(width=0, shape="hv"),
            fillcolor=None,  # self.fillcolor,
            fillgradient=dict(type="vertical", colorscale=color_scale),
            fill="tozeroy",
            showlegend=False,
            hoverinfo="skip",
            zorder=self.zorder,
            opacity=0.5
        )

        fig.add_trace(leverage, row=self.row, col=self.col)

        # draw a line (with shadow) on top of the fill area
        Line(
            label=self.label,
            column=self.column,
            color=self.color,
            shape='hv',
            width=max(self.width - 1, 0.1),
            fillcolor=None,
            fillmethod=None,
            row=self.row,
            col=self.col,
            end_marker=False,
            zorder=self.zorder + 1
        ).add_trace(fig, data)

        return fig


@dataclass
class PositionSize(ChartElement):
    fillcolor: Color | None = None
    fillmethod: str = "tozeroy"

    zorder: int = 0

    def apply_style(self, style: TikrStyle) -> None:
        self.width = style.line_width if not self.width else self.width
        self.color = self.color if self.color else style.colors.strategy
        self.fillcolor = self.fillcolor \
            if self.fillcolor else style.colors.strategy_fill

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        # Ensure column is numeric
        data[self.column] = pd.to_numeric(data[self.column], errors="coerce")

        if data[self.column].isna().any():
            logger.warning(
                f"NaN values found in {self.column}. Handling missing values."
                )
            data[self.column] = data[self.column].ffill()

        # Ensure data is sorted and aligned
        if data.index.duplicated().any():
            logger.warning("Duplicated x-values detected; sorting index.")
            data = data.sort_index()

        assert pd.api.types.is_numeric_dtype(data[self.column]), \
            f"Column {self.column} must be numeric!"

        Line(
            label=self.label,
            column=self.column,
            color=self.color,
            shape='linear',
            width=max(self.width - 1, 0.1),
            fillcolor=self.fillcolor,
            fillmethod=self.fillmethod,
            row=self.row,
            col=self.col,
            end_marker=False,
            zorder=self.zorder
        ).add_trace(fig, data)


@dataclass
class Signals(ChartElement):
    fillcolor: Color | None = None
    fillmethod: str = "tozeroy"

    zorder: int = 0

    def apply_style(self, style: TikrStyle) -> None:
        self.width = style.line_width if not self.width else self.width
        self.color = self.color if self.color else style.colors.strategy
        self.fillcolor = self.fillcolor \
            if self.fillcolor else style.colors.strategy_fill

    def add_trace(self, fig: go.Figure, data: pd.DataFrame) -> go.Figure:
        # Ensure column is numeric
        data[self.column] = pd.to_numeric(data[self.column], errors="coerce")

        if data[self.column].isna().any():
            logger.warning(
                f"NaN values found in {self.column}. Handling missing values."
                )
            data[self.column] = data[self.column].ffill()

        # Ensure data is sorted and aligned
        if data.index.duplicated().any():
            logger.warning("Duplicated x-values detected; sorting index.")
            data = data.sort_index()

        assert pd.api.types.is_numeric_dtype(data[self.column]), \
            f"Column {self.column} must be numeric!"

        Line(
            label=self.label,
            column=self.column,
            color=self.color,
            shape='hv',
            width=max(self.width - 1, 0.1),
            fillcolor=self.fillcolor,
            fillmethod=self.fillmethod,
            row=self.row,
            col=self.col,
            end_marker=False,
            zorder=self.zorder
        ).add_trace(fig, data)


# ====================================================================================
@dataclass
class SubPlot:
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
    elements: Sequence[tuple[str, str]] = field(default_factory=list)
    level: str = field(default="indicator")

    is_subplot: bool = False

    secondary_y: bool = False
    _row: int = 1
    _col: int = 1

    def __repr__(self) -> str:
        elements = (
            "elements:\n" + "\n".join(
                ["\t" + str(chan) for chan in self.elements]
                )
            if self.elements
            else ""
        )

        return "\n".join(
            (
                f"<<<<< SubPlot: {self.label} >>>>>\n",
                f"level='{self.level}'",
                f"is_subplot={self.is_subplot}",
                f"secondary_y={self.secondary_y})\n",
                f"{elements}\n\n",
            )
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self.elements)

    def __add__(self, other: "SubPlot") -> "SubPlot":
        res_levels = {
            0: "indicator",
            1: "operand",
            2: "condition",
            3: "signal generator",
        }

        # determine the 'level' of the new description after addition
        level = max(
            (
                k
                for k, v in res_levels.items()
                for candidate in (self.level, other.level)
                if v == candidate
            )
        )

        logger.debug(f"Combining {self.label} and {other.label}")

        elements = tuple(itertools.chain(self.elements, other.elements))

        # TODO: implement this part to plot channels between triggers
        # logger.info(elements)

        # # combine the lines
        # lines = [elem for elem in elements if not elem.is_trigger]

        # logger.info("LINES after filtering out triggers: %s" % lines)

        # # combine the triggers
        # triggers = [elem for elem in elements if elem.is_trigger]

        # if triggers:
        #     trig_channel = [
        #         elem[0]
        #         for pair in itertools.pairwise(triggers)
        #         if pair[0][0].split("_")[0] == pair[1][0].split("_")[0]
        #         for elem in pair
        #     ]
        # else:
        #     trig_channel = []

        # # combine the channels ... include the triggers, so we can
        # # plot a channel between 'overbought' and 'oversold' for example
        # channel = trig_channel

        # # sometimes the addition can cause a line to be both a line
        # # and a trigger, so we need to remove it from the lines
        # lines = [line for line in lines if line not in triggers]

        sub_plot = SubPlot(
            label=self.label,
            is_subplot=self.is_subplot & other.is_subplot,
            elements=elements,
            level=res_levels[min(level + 1, 3)],
        )

        logger.debug(sub_plot)

        return sub_plot

    def __radd__(self, other: "SubPlot") -> "SubPlot":
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __post_init__(self):
        # set the legendgroup for all elements in the subplot that 
        # do not have a legendgroup yet
        try:
            for elem in self.elements:
                elem.legendgroup = elem.legendgroup or self.label
                logger.debug(elem)
                logger.debug("-" * 150)
        except Exception as e:
            logger.error(f"Error setting legendgroup: {e}")
            logger.error(self.elements)

    @property
    def row(self) -> int:
        return self._row

    @row.setter
    def row(self, value: int):
        self._row = value
        for element in self.elements:
            element.row = value

    @property
    def col(self) -> int:
        return self._col

    @col.setter
    def col(self, value: int):
        self._col = value
        for element in self.elements:
            element.col = value

    def apply_style(self, style):
        for element in self.elements:
            element.apply_style(style)

    def draw_subplot(self, fig, data) -> go.Figure:
        """Draw the subplot and add a legend within the subplot"""
        for element in self.elements:
            logger.info(
                "[%s]adding trace to subplot: %s (column: %s)", 
                self.label, element.label, element.column
                )
            try:
                element.add_trace(fig, data)
            except Exception as e:
                logger.error(f"Error drawing subplot: {element.label}")
                logger.error(str(e), exc_info=True)
                logger.error(data.columns)

    def get_axis_keys(self, fig, row, col):
        """Get the axis keys for a specific subplot."""
        subplot_refs = fig._grid_ref

        ref = subplot_refs[row - 1][0][0]

        print(ref)

        return ref.layout_keys[0], ref.layout_keys[1]


@validate_layout(validation_level=VALIDATION_LEVEL)
@dataclass
class Layout:
    layout: Dict[str, Dict] = field(default_factory=dict)

    row_heights: List[float] = field(default_factory=list)
    col_widths: List[float] = field(default_factory=list)

    vertical_spacing: float = 0.05
    horizontal_spacing: float = 0.05

    @property
    def number_of_rows(self) -> int:
        if self.layout:
            return max((cell.get("row", 1) for cell in self.layout.values()))
        else:
            return 0

    @property
    def number_of_columns(self) -> int:
        return max((cell.get("col", 1) for cell in self.layout.values()))

    @property
    def rows(self) -> Tuple[Tuple[Dict]]:
        """Get all layout definitions in the layout as rows."""
        return tuple(
            {k: v for k, v in self.layout.items() if v.get("row") == row_number}
            for row_number in range(1, self.number_of_rows + 1)
        )

    @property
    def columns(self) -> Tuple[Tuple[Dict]]:
        """Get all layout definitions in the layout as rows."""
        cond = lambda x, column: x.get("row", None) == column  # noqa: E731

        return tuple(
            tuple(filter(lambda x: cond(x, row_number), self.layout.values()))
            for row_number in range(1, self.number_of_rows + 1)
        )

    def build_specs(self, subplots: List[SubPlot]) -> Specs:
        """Build the specifications for each subplot."""

        def build_spec(id_, elem: dict) -> Dict:
            """Builds a spec for a single subplot."""
            subplot = next(filter(lambda x: x.label == id_, subplots), None)

            return (
                dict(
                    row=elem.get("row"),
                    col=elem.get("col"),
                    rowspan=elem.get("rowspan", 1),
                    colspan=elem.get("colspan", 1),
                    name=subplot.label,
                    secondary_y=subplot.secondary_y,
                )
                if subplot
                else {}
            )

        def build_specs_for_row(row):
            """Builds specs for a single row of subplots."""
            items = (build_spec(key, elem) for key, elem in row.items())
            return tuple(sorted(items, key=lambda x: x["col"], reverse=False))

        return tuple(build_specs_for_row(row) for row in self.rows)

    def get_row(self, row_number: int) -> List[Dict]:
        """Get all subplots in a specific row."""
        if row_number < 1 or row_number > self.number_of_rows:
            raise ValueError(f"Row number {row_number} out of range.")

        return self.rows[row_number]

    def get_subplot_positions(self) -> Dict[str, Tuple[int, int]]:
        """Get the position of each subplot."""
        positions = {}
        for key, spec in self.layout.items():
            row = spec.get("row", 1)
            col = spec.get("col", 1)
            positions[key] = (row, col)
        return positions

    def show_layout(self) -> None:
        if not self.layout:
            raise ValueError("Layout is not defined/empty.")

        cell_width = 12

        # Find the maximum row and column, considering rowspan and colspan
        max_row = 0
        max_col = 0
        for spec in self.layout.values():
            row = spec.get("row", 1)
            col = spec.get("col", 1)
            rowspan = spec.get("rowspan", 1)
            colspan = spec.get("colspan", 1)
            max_row = max(max_row, row + rowspan - 1)
            max_col = max(max_col, col + colspan - 1)

        # Create an empty grid with extra space for borders
        grid = [
            [" " for _ in range(max_col * cell_width + 1)]
            for _ in range(max_row * 2 + 1)
        ]

        # Fill in the grid
        for subplot, spec in self.layout.items():
            row = spec.get("row", 1) - 1
            col = spec.get("col", 1) - 1
            rowspan = spec.get("rowspan", 1)
            colspan = spec.get("colspan", 1)

            # Draw the box
            for r in range(row * 2, (row + rowspan) * 2 + 1):
                for c in range(col * cell_width, (col + colspan) * cell_width + 1):
                    if r % 2 == 0:
                        grid[r][c] = "-"
                    elif c % cell_width == 0:
                        grid[r][c] = "|"

            # Add the subplot label
            label_row = row * 2 + 1
            label_col = col * cell_width + 2
            label = subplot[: colspan * cell_width - 2]  # Truncate label if too long
            grid[label_row][label_col: label_col + len(label)] = label

        # Print the grid
        for row in grid:
            print("".join(row))

        print("\nLayout specifications:")
        for subplot, spec in self.layout.items():
            print(f"{subplot}: {spec}")

        print("\n")
        print(f"Number of rows: {self.number_of_rows}")
        print(f"Number of columns: {self.number_of_columns}")
        print(f"Row heights: {self.row_heights}")
        print(f"Column widths: {self.col_widths}")


@dataclass(frozen=True)
class PlotDefinition:
    title: str
    subplots: Sequence[SubPlot]
    layout: Layout
    style: TikrStyle

    def __repr__(self) -> str:
        subplots = "\n" + "\n".join(
            [
                "" + "\n".join(["" + str(sub) for sub in subplot])
                for subplot in self.subplots
            ]
        )

        return "\n".join(
            (
                f"PlotDefinition name: {self.title}\n",
                f"style={self.style}\n",
                f"[{self.number_of_subplots}] subplots:\n{subplots}\n",
            )
        )

    def __post_init__(self):
        positions = self.layout.get_subplot_positions()

        for subplot in self.subplots:
            try:
                subplot.row, subplot.col = positions[subplot.label]
            except KeyError:
                raise ValueError(f"Subplot '{subplot.label}' not found in layout.")

        for element in self.elements:
            element.apply_style(self.style)

    @property
    def number_of_subplots(self) -> int:
        """Calculate the number of subplots needed for the plot."""
        return len(self.subplots)

    @property
    def subplot_titles(self) -> List[str]:
        return [subplot.label for subplot in self.subplots]

    @property
    def specs(self) -> Specs:
        return self.layout.build_specs(self.subplots)

    @property
    def elements(self) -> List[ChartElement]:
        return [elem for subplot in self.subplots for elem in subplot.elements]

    @contextmanager
    def style_context(self):
        previous_style = getattr(PlotDefinition, "_current_style", None)
        PlotDefinition._current_style = self.style
        try:
            yield
        finally:
            PlotDefinition._current_style = previous_style

    @staticmethod
    def get_current_style():
        return getattr(PlotDefinition, "_current_style", None)

    def apply_style(self, element):
        style = self.get_current_style()
        if style:
            element.apply_style(style)


if __name__ == "__main__":
    lines = [
        Line(
            label=f"Line {int(random() * 100)}",
            column=choice(["open", "high", "low", "close", "volume"]),
        )
        for _ in range(100)
    ]

    channels = [
        Channel(
            label=f"Channel {int(random() * 100)}",
            upper=choice(lines),
            lower=choice(lines),
        )
        for _ in range(100)
    ]

    plot_definition = PlotDefinition(
        name="Test Plot",
        subplots=[
            SubPlot(label="1-1", elements=[choice(lines) for _ in range(3)]),
            SubPlot(label="2-1", elements=[choice(lines) for _ in range(2)]),
            SubPlot(label="2-2", elements=[channels[0]]),
            SubPlot(label="3-1", elements=[channels[1]]),
        ],
        layout=Layout(
            {
                "1-1": {"row": 1, "col": 1, "rowspan": 2, "colspan": 3},
                "2-2": {"row": 3, "col": 2, "colspan": 2},
                "2-1": {"row": 3, "col": 1},
                "3-1": {"row": 4, "col": 1, "colspan": 3},
            }
        ),
        style=None,
    )

    # print('=' * 150)
    # print(plot_definition)
    # print('=' * 150)
    logger.info("number of subplots %s" % plot_definition.number_of_subplots)

    layout = plot_definition.layout

    logger.info(
        "rows: %s / columns: %s",
        layout.number_of_rows,
        layout.number_of_columns,
    )

    plot_definition.layout.show_layout()

    for row in plot_definition.specs:
        print(row)

    cs = Candlestick(
        label="OHLCV", color_up=Color(255, 0, 0, 1), color_down=Color(0, 255, 0, 1)
    )

    print(cs)
