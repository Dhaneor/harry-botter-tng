#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 21:55:20 2024

@author dhaneor
"""
from dataclasses import dataclass
from typing import Sequence
from pprint import pprint


class Color:
    def __init__(self, red=0, green=0, blue=0, alpha=1.0):
        self.r: int = red
        self.g: int = green
        self.b: int = blue
        self.a: float = alpha

    def __str__(self) -> str:
        return self.rgba

    def __repr__(self) -> str:
        return self.rgba

    def set_alpha(self, alpha: float) -> None:
        self.a = alpha
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

        color = cls(*hex_to_rgb(hex_color))
        color.set_alpha(alpha)
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

        self.canvas = Color.from_hex(self.canvas)
        self.background = Color.from_hex(self.background)
        self.grid = Color.from_hex(self.grid)
        self.text = Color.from_hex(self.text)

        pprint(self)

    def add_fill_colors(self, fill_alpha) -> None:
        print("adding fill colors with alpha", fill_alpha)
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
            canvas=palette[7],
            background=palette[8],
            grid=palette[9],
            text=palette[10],
        )


@dataclass
class TikrStyle:
    colors: Colors
    line_width: float = 1

    line_alpha: float = 0.75
    fill_alpha: float = 0.4
    shadow_alpha: float = 0.2

    candle_up_alpha: float = 1
    candle_down_alpha: float = 1

    font_family: str = "Arial"
    font_size: int = 12

    def __post_init__(self):
        print("running color updates ...")
        self.colors.add_fill_colors(self.fill_alpha)
        self.colors.update_line_alpha(self.line_alpha)


# ==================================== tikr styles ===================================
palette_1 = [
    "#f19c79",
    "#fbdcbd",
    "#a44a3f",
    "#57ae1a",
    "#a42f0e",
    "#6cef0f",
    "#e8491d",
    "#eae7de",
    "#cdc3a9",
    "#BBBBBB",
    "#636363",
]

palette_2 = [
    "#0d7b3e",
    "#2ae088",
    "#e74c3c",
    "#156d3a",
    "#9b59b6",
    "#34495e",
    "#e67e22",
    "#c3caca",
    "#ecf0f1",
    "#f9f9f9",
    "#3f3f3f",
]

palette_3 = [
    "#2A9D8F",
    "#3fde54",
    "#E76F51",
    "#2ecc71",
    "#9e2a2b",
    "#57ae1a",
    "#e8491d",
    "#540b0e",
    "#794a3a",
    "#9b59b6",
    "#636363",
]

tikr_day_colors = Colors.from_palette(palette_2)

tikr_day_colors = Colors(
    strategy="#0d7b3e",
    capital="#2ae088",
    hodl="#e74c3c",
    candle_up="#156d3a",
    candle_down="#9b59b6",
    buy="#34495e",
    sell="#e67e22",
    canvas="#c3caca",
    background="#ecf0f1",
    grid="#f9f9f9",
    text="#3f3f3f",
)

tikr_day_style = TikrStyle(
    colors=tikr_day_colors,
    line_width=1,
    line_alpha=0.75,
    fill_alpha=0.4,
    shadow_alpha=0.2,
    candle_up_alpha=1,
    candle_down_alpha=1,
    font_family="Arial",
    font_size=12,
)

tikr_day_colors = Colors(
    strategy="#2c7a4a",  # Rich forest green
    capital="#56c17e",  # Vivid mint green
    hodl="#d9544f",  # Soft crimson red
    candle_up="#1f804d",  # Slightly brighter green
    candle_down="#884ea0",  # Muted plum
    buy="#5d6d7e",  # Steel blue
    sell="#d35400",  # Warm amber
    canvas="#e8e9eb",  # Light warm gray
    background="#f7f9fb",  # Soft off-white
    grid="#d6d6d6",  # Subtle light gray
    text="#2d2d2d",  # Darker gray
)

tikr_day_style = TikrStyle(
    colors=tikr_day_colors,
    line_width=1,
    line_alpha=0.8,  # Slightly more opaque lines
    fill_alpha=0.5,  # More transparent fills
    shadow_alpha=0.3,  # Slightly stronger shadow
    candle_up_alpha=1,  # Keep full opacity for candles
    candle_down_alpha=1,
    font_family="Arial",
    font_size=12,
)

tikr_day_colors = Colors(
    strategy="#f39c12",  # Warm muted orange
    capital="#f4d03f",  # Soft golden yellow
    hodl="#d9544f",  # Soft crimson red
    candle_up="#f39c12",  # Matching strategy for positive candles
    candle_down="#884ea0",  # Muted plum
    buy="#5d6d7e",  # Steel blue
    sell="#d35400",  # Warm amber
    canvas="#e8e9eb",  # Light warm gray
    background="#f7f9fb",  # Soft off-white
    grid="#d6d6d6",  # Subtle light gray
    text="#a8a8ab",  # Darker gray
)

tikr_day_style = TikrStyle(
    colors=tikr_day_colors,
    line_width=1.5,
    line_alpha=1,  # Slightly more opaque lines
    fill_alpha=0.3,  # More transparent fills
    shadow_alpha=0.3,  # Slightly stronger shadow
    candle_up_alpha=1,  # Keep full opacity for candles
    candle_down_alpha=1,
    font_family="Arial",
    font_size=12,
)

# ==============================' night colors/styles ================================
palette_4 = [
    "#d87c26",
    "#f7b538",
    "#c32f27",
    "#2ecc71",
    "#9e2a2b",
    "#57ae1a",
    "#e8491d",
    "#181818",
    "#202020",
    "#121212",
    "#7f818a",
]

tikr_night_colors = Colors.from_palette(palette_4)
tikr_night_style = TikrStyle(colors=tikr_night_colors)

# =============================== build styles dictionary ============================
styles = {
    "default": tikr_day_style,  # default style for tikr charting
    "day": tikr_day_style,
    "night": tikr_night_style,
    # add more styles here...
}


if __name__ == "__main__":
    color = Color.from_hex("#d4e09b", 0.5)
    print(color)

    print(f"Hex color: {color.hex}")
    print(f"RGBA color string: {color.rgba}")
    print(f"RGBA tuple (0-1 alpha): {color.rgba_tuple}")

    pprint(tikr_day_style.colors)
