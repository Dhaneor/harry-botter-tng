#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 21:55:20 2024

@author dhaneor
"""
from dataclasses import dataclass
from typing import Sequence


class Color:
    def __init__(self, hex_color: str, alpha: float = 1.0):
        self._hex_color = hex_color
        self.alpha = alpha

    def __str__(self) -> str:
        return self.rgba

    def __repr__(self) -> str:
        return f"{self.rgba}"

    @property
    def hex(self) -> str:
        return self._hex_color

    @property
    def rgba(self) -> str:
        r, g, b, a = self.to_rgba_tuple()
        return f"rgba({r}, {g}, {b}, {a})"

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def to_rgba_tuple(self) -> tuple[int, int, int, float]:
        r, g, b = self._hex_to_rgb(self._hex_color)
        return (r, g, b, self.alpha)

    def to_rgba_255(self) -> tuple[int, int, int, int]:
        r, g, b = self._hex_to_rgb(self._hex_color)
        return (r, g, b, int(self.alpha * 255))


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

    strategy_fill: Color = None
    capital_fill: Color = None
    hodl_fill: Color = None

    def __post_init__(self):
        # turn the colors into Color objects
        self.strategy = Color(self.strategy)
        self.capital = Color(self.capital)
        self.hodl = Color(self.hodl)
        self.candle_up = Color(self.candle_up)
        self.candle_down = Color(self.candle_down)
        self.buy = Color(self.buy)
        self.sell = Color(self.sell)

        self.canvas = Color(self.canvas)
        self.background = Color(self.background)
        self.grid = Color(self.grid)
        self.text = Color(self.text)

    def add_fill_colors(self, fill_alpha) -> None:
        self.strategy_fill = Color(self.strategy.hex, fill_alpha)
        self.capital_fill = Color(self.capital.hex, fill_alpha)
        self.hodl_fill = Color(self.hodl.hex, fill_alpha)

    def update_line_alpha(self, alpha: float) -> None:
        self.strategy.alpha = alpha
        self.capital.alpha = alpha
        self.hodl.alpha = alpha
        self.candle_up.alpha = alpha
        self.candle_down.alpha = alpha
        self.buy.alpha = alpha
        self.sell.alpha = alpha
        self.canvas.alpha = alpha
        self.background.alpha = alpha
        self.grid.alpha = alpha
        self.text.alpha = alpha

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
            text=palette[10]
        )


@dataclass
class TikrStyle:
    colors: Colors
    line_width: float = 0.75

    line_alpha: float = 0.75
    fill_alpha: float = 0.2
    shadow_alpha: float = 0.2

    candle_up_alpha: float = 1
    candle_down_alpha: float = 1

    font: str = "Arial"
    font_size: int = 10

    def __post_init__(self):
        self.colors.add_fill_colors(self.fill_alpha)


# ==================================== tikr styles ===================================
palette_1 = [
    "#196630", "#1b53a6", "#a44a3f",
    "#57ae1a", "#a42f0e", "#6cef0f", "#e8491d",
    "#eae7de", "#cdc3a9", "#BBBBBB", "#636363"
]

tikr_day_colors = Colors.from_palette(palette_1)
tikr_day_style = TikrStyle(colors=tikr_day_colors)

styles = {
    "default": tikr_day_style,  # default style for tikr charting
    "day": tikr_day_style,
    # add more styles here...
}


if __name__ == "__main__":
    color = Color("#d4e09b", 0.5)
    print(f"Hex color: {color.hex}")
    print(f"RGBA color string: {color.rgba}")
    print(f"RGBA tuple (0-1 alpha): {color.to_rgba_tuple()}")
    print(f"RGBA tuple (0-255 alpha): {color.to_rgba_255()}")
    print(tikr_day_style)
