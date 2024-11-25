#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
import pandas as pd
import plotly.graph_objects as go
import sys

from collections import OrderedDict
from itertools import chain
from pprint import pprint
from random import random, choice
from typing import Sequence, List, Dict, Tuple
from dataclasses import dataclass, field

from layout_validator import validate_layout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a StreamHandler to output log messages to the console
handler = logging.StreamHandler(sys.stdout)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

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


# ========================== Classes for chart components ============================
@dataclass
class Line:
    label: str | None = None
    column: str | None = None
    legendgroup: str | None = None
    color: Color | None = None
    width: float = 1
    opacity: float = 1
    shape: str = "spline"
    zorder: int | None = 0

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
            color = Color(
                self.color.r / factor,
                self.color.g / factor,
                self.color.b / factor,
                self.color.a / factor).rgba  # replace with lighter color

            line = dict(color=color, width=self.width + factor, shape=self.shape)

            shadow_trace = go.Scatter(
                x=data.index,
                y=data[self.column or self.label],
                line=line,
                opacity=self.opacity,
                hoverinfo='skip',
                showlegend=False,
                zorder=self.zorder - factor,
            )
            fig.add_trace(shadow_trace, row=row, col=col)

        return fig

    def as_dict(self) -> dict:
        return {
            "color": self.color.rgba,
            "width": self.width,
            "shape": self.shape,
        }


class Trigger(Line):
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

    _legendgroup: str | None = None

    def __repr__(self) -> str:
        return (
            f"Channel {self.label}\n\t\tupper: {self.upper}\n\t\tlower: {self.lower}"
        )

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

    def __post_init__(self):
        for line in [self.upper, self.lower]:
            match line:
                case str():
                    line = Line(label=line, column=line)
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
    _legendgroup: str | None = None
    line: Line | str | float | None = None  # line for the dd levels
    color: Color | None = None
    gradient: bool = True,
    critical: float | None = None
    opacity: float = 1
    color_scale: list[tuple[float, str]] | None = None

    @property
    def legendgroup(self) -> str:
        return self._legendgroup or self.label

    @legendgroup.setter
    def legendgroup(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"Invalid legendgroup: {value}")

        self._legendgroup = value
        self.line.legendgroup = value

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
    lines: Sequence[tuple[str, str]] = field(default_factory=list)
    triggers: Sequence[tuple[str, str]] = field(default_factory=list)
    channels: Sequence[str] = field(default_factory=list)
    hist: Sequence[str] = field(default_factory=list)
    level: str = field(default="indicator")
    is_subplot: bool = True
    secondary_y: bool = False

    def __repr__(self) -> str:
        lines = 'lines:\n' + '\n'.join(['\t' + str(line) for line in self.lines])\
            if self.lines else ''
        triggers = 'triggers:\n' + '\n'\
            .join(['\t' + str(trigger) for trigger in self.triggers])\
            if self.triggers else ''
        hist = 'hist:\n' + '\n'\
            .join(['\t' + str(hist) for hist in self.hist])\
            if self.hist else ''
        channels = 'channels:\n' + '\n'\
            .join(['\t' + str(chan) for chan in self.channels])\
            if self.channels else ''

        return "\n".join((
            f"<<<<< SubPlot {self.label} >>>>>\n",
            f"level='{self.level}'",
            f"is_subplot={self.is_subplot}",
            f"secondary_y_axis={self.secondary_y_axis})\n",
            f"{lines}{triggers}{channels}{hist}\n\n",
        ))

    def __post_init__(self):
        for idx in range(len(self.lines)):
            line = self.lines[idx]
            if isinstance(line, tuple) and len(line) == 2:
                self.lines[idx] = Line(label=self.label)

            logger.debug(line)

        logger.debug('-' * 150)

        # set all elements contained in this subplot to have the
        # same legendgroup
        for elem in chain(self.lines, self.triggers, self.channels, self.hist):
            logger.debug("setting legendgroup for %s" % elem.label)
            elem.legendgroup = self.label
            logger.debug(elem)
            logger.debug('-' * 150)


@validate_layout(validation_level=VALIDATION_LEVEL)
@dataclass
class Layout:
    layout: Dict[str, Dict] = field(default_factory=dict)

    @property
    def number_of_rows(self) -> int:
        return max((cell.get('row', 1) for cell in self.layout.values()))

    @property
    def number_of_columns(self) -> int:
        return max((cell.get('col', 1) for cell in self.layout.values()))

    @property
    def rows(self) -> Tuple[Tuple[Dict]]:
        """Get all layout definitions in the layout as rows."""
        return tuple(
            {k: v for k, v in self.layout.items() if v.get('row') == row_number}
            for row_number in range(1, self.number_of_rows + 1)
        )

    @property
    def columns(self) -> Tuple[Tuple[Dict]]:
        """Get all layout definitions in the layout as rows."""
        cond = lambda x, column: x.get('row', None) == column  # noqa: E731

        return tuple(
            tuple(filter(lambda x: cond(x, row_number), self.layout.values()))
            for row_number in range(1, self.number_of_rows + 1)
        )

    def build_specs(self, subplots: List[SubPlot]) -> Specs:
        """Build the specifications for each subplot."""

        def build_spec(id_, elem: dict) -> Dict:
            """Builds a spec for a single subplot."""
            subplot = next(
                filter(lambda x: x.label == id_, subplots),
                None
            )

            return OrderedDict(
                row=elem.get('row'),
                col=elem.get('col'),
                rowspan=elem.get('rowspan', 1),
                colspan=elem.get('colspan', 1),
                name=subplot.label,
                secondary_y=subplot.secondary_y
            ) if subplot else OrderedDict()

        def build_specs_for_row(row):
            """Builds specs for a single row of subplots."""
            return tuple(build_spec(key, elem) for key, elem in row.items())

        return tuple(build_specs_for_row(row) for row in self.rows)

    def get_row(self, row_number: int) -> List[Dict]:
        """Get all subplots in a specific row."""
        if row_number < 1 or row_number > self.number_of_rows:
            raise ValueError(f"Row number {row_number} out of range.")

        return self.rows[row_number]

    def get_subplot_position(self) -> Dict[str, Tuple[int, int]]:
        """Get the position of each subplot."""
        positions = {}
        for key, spec in self.layout.items():
            row = spec.get('row', 1)
            col = spec.get('col', 1)
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
            row = spec.get('row', 1)
            col = spec.get('col', 1)
            rowspan = spec.get('rowspan', 1)
            colspan = spec.get('colspan', 1)
            max_row = max(max_row, row + rowspan - 1)
            max_col = max(max_col, col + colspan - 1)

        # Create an empty grid with extra space for borders
        grid = [
            [' ' for _ in range(max_col * cell_width + 1)]
            for _ in range(max_row * 2 + 1)
        ]

        # Fill in the grid
        for subplot, spec in self.layout.items():
            row = spec.get('row', 1) - 1
            col = spec.get('col', 1) - 1
            rowspan = spec.get('rowspan', 1)
            colspan = spec.get('colspan', 1)

            # Draw the box
            for r in range(row * 2, (row + rowspan) * 2 + 1):
                for c in range(col * cell_width, (col + colspan) * cell_width + 1):
                    if r % 2 == 0:
                        grid[r][c] = '-'
                    elif c % cell_width == 0:
                        grid[r][c] = '|'

            # Add the subplot label
            label_row = row * 2 + 1
            label_col = col * cell_width + 2
            label = subplot[:colspan * cell_width - 2]  # Truncate label if too long
            grid[label_row][label_col:label_col + len(label)] = label

        # Print the grid
        for row in grid:
            print(''.join(row))

        print("\nLayout specifications:")
        for subplot, spec in self.layout.items():
            print(f"{subplot}: {spec}")


@dataclass(frozen=True)
class PlotDefinition:
    name: str
    subplots: Sequence[Sequence[SubPlot]]
    layout: Layout
    style: TikrStyle = None

    def __repr__(self) -> str:
        subplots = '\n' + '\n'.join(
            [
                '' + '\n'.join(['' + str(sub) for sub in subplot])
                for subplot in self.subplots
            ]
        )

        return "\n".join((
            f"PlotDefinition name: {self.name}\n",
            f"style={self.style}\n",
            f"[{self.number_of_subplots}] subplots:\n{subplots}\n",
        ))

    @property
    def number_of_subplots(self) -> int:
        """Calculate the number of subplots needed for the plot."""
        return len(self.subplots)

    @property
    def specs(self) -> Specs:
        return self.layout.build_specs(self.subplots)


if __name__ == "__main__":
    lines = [
        Line(
            label=f"Line {int(random() * 100)}",
            column=choice(["open", "high", "low", "close", "volume"])
        ) for _ in range(100)
    ]

    channels = [
        Channel(
            label=f"Channel {int(random() * 100)}",
            upper=choice(lines),
            lower=choice(lines),
        ) for _ in range(100)
    ]

    plot_definition = PlotDefinition(
        name="Test Plot",
        subplots=[

                SubPlot(label="1-1", lines=[choice(lines) for _ in range(3)]),
                SubPlot(label="1-2", lines=[("Value", "close")]),
                SubPlot(label="2-1", channels=[channels[0]]),
                SubPlot(label="2-2", channels=[channels[1]]),
        ],
        layout=Layout(
            {
                "1-1": {"row": 1, "col": 1, "rowspan": 2, "colspan": 3},
                "1-2": {"row": 2, "col": 1},
                "2-1": {"row": 2, "col": 2},
                "2-2": {"row": 4, "col": 1, "colspan": 2},
            }
        )
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
    logger.info(plot_definition.layout.rows)

    for row in plot_definition.specs:
        print(row)

    # print(plot_definition.layout.build_specs(plot_definition.subplots))
