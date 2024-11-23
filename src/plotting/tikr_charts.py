#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import io
import logging

from abc import abstractmethod
from plotly.io import to_image

from src.analysis.strategies import strategy_plot as sp

# from src.analysis.models import positions as pos
from src.analysis.indicators.indicator import PlotDescription
from src.plotting.plotly_styles import styles, TikrStyle

logger = logging.getLogger(f"main.{__name__}")


# ============================= Define plot descriptions =============================
class TikrChartBase:
    def __init__(self, df, style, title=None):
        self.data = self._prepare_data(df.copy())
        self.title = title

        self.style: TikrStyle
        self._set_style(style)

        self._plotdefinition: sp.PlotDefinition = None

    @property
    def plot_definition(self):
        return self._plot_definition

    @abstractmethod
    def draw(self): ...

    def get_image_bytes(
        self, format="png", scale=3, width=1200, height=800
    ) -> io.BytesIO:
        """
        Generate a high-quality BytesIO object containing the chart image.

        Args:
            format (str): Image format ('png', 'jpeg', 'webp', 'svg', 'pdf')
            scale (int): Scale factor for the image (default 3 for high quality)
            width (int): Base width of the image in pixels (default 1200)
            height (int): Base height of the image in pixels (default 800)

        Returns:
            io.BytesIO: BytesIO object containing the high-quality image
        """
        fig = sp.build_figure(data=self.data, p_def=self.plot_definition)

        img_bytes = to_image(
            fig,
            format=format,
            scale=scale,
            width=width,
            height=height
        )

        return io.BytesIO(img_bytes)

    # ................................. Helper Methods ...............................
    @abstractmethod
    def _build_plot_definition(self): ...

    def _prepare_data(self, data):
        for col in data.columns:
            data[col] = data[col] * -100 if "drawdown" in col else data[col]
        return data

    def _set_style(self, style: str):
        if style not in styles:
            logger.warning(f"Style '{style}' not found. Using default style.")
            style = "default"

        self.style = styles[style]


class TikrChart(TikrChartBase):
    def __init__(self, df, style: str, title=None):
        super().__init__(df, style, title)
        self._plot_definition = self._build_plot_definition()

    def draw(self):
        sp.plot(data=self.data, p_def=self.plot_definition)

    def _build_plot_definition(self):
        # ...................... Line/Fill Area Definitions ..........................
        hodl_drawdown_fill_area = sp.Drawdown(
            label="HODL drawdown",
            column="hodl.drawdown",
            line=sp.Line(
                label="hodl.drawdown",
                color=self.style.colors.hodl,
                width=self.style.line_width,
            ),
            color=self.style.colors.hodl_fill,  # fill_color
            opacity=0.1,
        )

        print(hodl_drawdown_fill_area)

        strategy_drawdown_fill_area = sp.Channel(
            label="b.drawdown",
            lower=sp.Line(
                label="b.drawdown",
                color=self.style.colors.strategy,
                width=self.style.line_width,
            ),
            color=self.style.colors.strategy_fill,
            opacity=0.1,
        )

        capital_drawdown_fill_area = sp.Channel(
            label="cptl.drawdown",
            lower=sp.Line(
                label="cptl.drawdown",
                color=self.style.colors.capital,
                width=self.style.line_width,
            ),
            color=self.style.colors.capital_fill,  # fill_color
            opacity=0.1,
        )

        # ............................ Plot Descriptions .............................
        drawdown = PlotDescription(
            label="Drawdown",
            is_subplot=True,
            lines=[],
            triggers=[],
            channel=[
                hodl_drawdown_fill_area,
                strategy_drawdown_fill_area,
                capital_drawdown_fill_area,
            ],
            hist=[],
            level="indicator",
        )

        portfolio = PlotDescription(
            label="Portfolio",
            is_subplot=True,
            lines=[
                sp.Line(
                    label="hodl.value",
                    color=self.style.colors.hodl,
                    width=self.style.line_width,
                ),
                sp.Line(
                    label="cptl.b",
                    color=self.style.colors.capital,
                    width=self.style.line_width,
                ),
                sp.Line(
                    label="b.value",
                    color=self.style.colors.strategy,
                    width=self.style.line_width,
                ),
            ],
            triggers=[],
            channel=[],
            hist=[],
            level="indicator",
        )

        # .............................. Define Plot Definition ......................
        return sp.PlotDefinition(
            name="Tikr Chart",
            main=None,
            sub=(
                drawdown,
                portfolio,
            ),
            style=self.style,
        )
