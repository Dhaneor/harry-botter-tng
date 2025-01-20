#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 21:55:20 2024

@author dhaneor
"""
from .plot_definition import Color, Colors, TikrStyle


# ==================================== tikr styles ===================================
# palette_1 = [
#     "#f19c79",
#     "#fbdcbd",
#     "#a44a3f",
#     "#57ae1a",
#     "#a42f0e",
#     "#6cef0f",
#     "#e8491d",
#     "#eae7de",
#     "#cdc3a9",
#     "#BBBBBB",
#     "#636363",
# ]

# palette_2 = [
#     "#0d7b3e",
#     "#2ae088",
#     "#e74c3c",
#     "#156d3a",
#     "#9b59b6",
#     "#34495e",
#     "#e67e22",
#     "#c3caca",
#     "#ecf0f1",
#     "#f9f9f9",
#     "#3f3f3f",
# ]

# palette_3 = [
#     "#2A9D8F",
#     "#3fde54",
#     "#E76F51",
#     "#2ecc71",
#     "#9e2a2b",
#     "#57ae1a",
#     "#e8491d",
#     "#540b0e",
#     "#794a3a",
#     "#9b59b6",
#     "#636363",
# ]

# tikr_day_colors = Colors.from_palette(palette_2)

# tikr_day_colors = Colors(
#     strategy="#0d7b3e",
#     capital="#2ae088",
#     hodl="#e74c3c",
#     candle_up="#156d3a",
#     candle_down="#9b59b6",
#     buy="#34495e",
#     sell="#e67e22",
#     canvas="#c3caca",
#     background="#ecf0f1",
#     grid="#f9f9f9",
#     text="#3f3f3f",
# )

# tikr_day_style = TikrStyle(
#     colors=tikr_day_colors,
#     line_width=1,
#     line_alpha=0.75,
#     fill_alpha=0.4,
#     shadow_alpha=0.2,
#     candle_up_alpha=1,
#     candle_down_alpha=1,
#     font_family="Arial",
#     font_size=12,
# )

# tikr_day_colors = Colors(
#     strategy="#2c7a4a",  # Rich forest green
#     capital="#56c17e",  # Vivid mint green
#     hodl="#d9544f",  # Soft crimson red
#     candle_up="#1f804d",  # Slightly brighter green
#     candle_down="#884ea0",  # Muted plum
#     buy="#5d6d7e",  # Steel blue
#     sell="#d35400",  # Warm amber
#     canvas="#e8e9eb",  # Light warm gray
#     background="#f7f9fb",  # Soft off-white
#     grid="#d6d6d6",  # Subtle light gray
#     text="#2d2d2d",  # Darker gray
# )

# tikr_day_style = TikrStyle(
#     colors=tikr_day_colors,
#     line_width=1,
#     line_alpha=0.8,  # Slightly more opaque lines
#     fill_alpha=0.5,  # More transparent fills
#     shadow_alpha=0.3,  # Slightly stronger shadow
#     candle_up_alpha=1,  # Keep full opacity for candles
#     candle_down_alpha=1,
#     font_family="Arial",
#     font_size=12,
# )

# tikr_day_colors = Colors(
#     strategy="#6fc351",  # Warm muted orange
#     capital="#51a8c3",  # Soft golden yellow "#f4d03f"
#     hodl="#c36c51",  # Soft crimson red
#     candle_up="#f39c12",  # Matching strategy for positive candles
#     candle_down="#884ea0",  # Muted plum
#     buy="#5d6d7e",  # Steel blue
#     sell="#d35400",  # Warm amber
#     canvas="#e8e9eb",  # Light warm gray
#     background="#f7f9fb",  # Soft off-white
#     grid="#d6d6d6",  # Subtle light gray
#     text="#a8a8ab",  # Darker gray
# )

# tikr_day_colors = Colors(
#     strategy="#8fd94f",  # Warm muted orange
#     capital="#4fd4d9",  # Soft golden yellow "#f4d03f"
#     hodl="#964148",  # Soft crimson red
#     candle_up="#f39c12",  # Matching strategy for positive candles
#     candle_down="#884ea0",  # Muted plum
#     buy="#5d6d7e",  # Steel blue
#     sell="#d35400",  # Warm amber
#     canvas="#ece9d6",  # Light warm gray
#     background="#ecded6",  # Soft off-white
#     grid="#d6d6d6",  # Subtle light gray
#     text="#a8a8ab",  # Darker gray
# )

# tikr_day_style = TikrStyle(
#     colors=tikr_day_colors,
#     line_width=1.5,
#     line_alpha=1,  # Slightly more opaque lines
#     fill_alpha=0.3,  # More transparent fills
#     shadow_alpha=0.3,  # Slightly stronger shadow
#     candle_up_alpha=1,  # Keep full opacity for candles
#     candle_down_alpha=1,
#     font_family="Arial",
#     font_size=12,
# )

tikr_day_colors = Colors(
    strategy="#27ae60",  # Lush emerald green for strategy
    capital="#2980b9",  # Muted cobalt blue for capital
    hodl="#c0392b",  # Deep crimson red for HODL
    candle_up="#2ecc71",  # Bright green for positive candles
    candle_down="#e67e22",  # Softened red for negative candles
    buy="#34495e",  # Steely blue for buy markers
    sell="#e74c3c",  # Amber for sell markers
    volume="#3498db",  # Light blue for volume bars
    canvas="#efe0ef",  # Neutral beige-gray for chart area
    background="#e8e9e9",  # Light warm white for outer background
    grid="#fff0ff",  # Subtle pale gray for grid lines
    text="#623c10",  # Dark slate gray for text
)

tikr_day_style = TikrStyle(
    colors=tikr_day_colors,
    line_width=1,
    line_alpha=0.5,  # Slightly more opaque lines for clarity
    fill_alpha=0.2,  # Balanced transparency for fills
    shadow_alpha=0.25,  # Subtle shadow
    candle_up_alpha=1,  # Full opacity for positive candles
    candle_down_alpha=1,  # Full opacity for negative candles
    font_family="Georgia, Arial",
    font_size=36,
    title_font_size=16,
    tick_font_size=16,
    font_opacity=0.5,
    volume_opacity=0.2,
    marker_size=10,
    marker_opacity=1,
    canvas_image="./assets/chart_bg.jpg",
    canvas_image_opacity=0.35
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
    "#404040",
    "#181818",
    "#202020",
    "#121212",
    "#999ba6",
]

tikr_night_colors = Colors.from_palette(palette_4)
tikr_night_style = TikrStyle(
    colors=tikr_night_colors,
    candle_up_alpha=0.6,
    candle_down_alpha=0.6,
    font_family="Fira Code, Console, monospace",
    font_size=24,
    title_font_size=16,
    tick_font_size=12,
    font_opacity=0.6,
    volume_opacity=0.3,
    marker_size=5,
    marker_opacity=1,
    canvas_image="./assets/chart_bg_noise.png",
    canvas_image_opacity=0.2
    )

backtest_style = TikrStyle(
    colors=tikr_night_colors,
    candle_up_alpha=0.7, 
    candle_down_alpha=0.7,
    line_width=0.3,
    line_alpha=0.75,
    fill_alpha=0.1,
    font_family="Fira Code, Console, monospace",
    font_size=10,
    title_font_size=12,
    tick_font_size=10,
    font_opacity=0.6,
    volume_opacity=0.3,
    marker_size=5,
    marker_opacity=1,
    canvas_image="./assets/chart_bg_noise.png",
    canvas_image_opacity=0.2
    )

# =============================== build styles dictionary ============================
styles = {
    "default": tikr_day_style,  # default style for tikr charting
    "day": tikr_day_style,
    "night": tikr_night_style,
    "backtest": backtest_style,
    # add more styles here...
}


if __name__ == "__main__":
    color = Color.from_hex("#d4e09b", 0.5)
    print(color)

    print(f"Hex color: {color.hex}")
    print(f"RGBA color string: {color.rgba}")
    print(f"RGBA tuple (0-1 alpha): {color.rgba_tuple}")

    # pprint(tikr_day_style.colors)

    print(color)
    print(color.set_alpha(1.0))
    print(color.reset())
