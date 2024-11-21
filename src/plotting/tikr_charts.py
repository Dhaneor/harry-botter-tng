#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
from abc import abstractmethod

from src.analysis.strategies import strategy_plot as sp
# from src.analysis.models import positions as pos
from src.analysis.indicators.indicator import PlotDescription
from src.plotting.plotly_styles import styles, TikrStyle

logger = logging.getLogger(f"main.{__name__}")

FILL_ALPHA = 0.2
DEFAULT_LINE_WIDTH = 0.75
DEFAULT_LINE_ALPHA = 0.75


# ============================= Define plot descriptions =============================
class TikrChartBase:
    def __init__(self, df, style, title=None):
        self.data = self.prepare_data(df.copy())
        self.title = title

        self.style: TikrStyle
        self._set_style(style)

        self._plotdefinition: sp.PlotDefinition = None

    @property
    def plot_definition(self):
        return self._plot_definition

    @abstractmethod
    def draw(self):
        ...

    @abstractmethod
    def _build_plot_definition(self):
        ...

    def prepare_data(self, data):
        for col in data.columns:
            data[col] = data[col] * -1 if "drawdown" in col else data[col]
        return data

    def _set_style(self, style: str):
        if style not in styles:
            logger.error(f"Style '{style}' not found. Using default style.")
            self.style = styles["default"]

        self.style = styles[style]


class TikrChart(TikrChartBase):
    def __init__(self, df, style: str, title=None):
        super().__init__(df, style, title)
        self._plot_definition = self._build_plot_definition()

    def draw(self):
        sp.plot(data=self.data, p_def=self.plot_definition)

    def _build_plot_definition(self):
        # ...................... Line/Fill Area Definitions ..........................
        hodl_drawdown_fill_area = sp.ChannelDefinition(
            label="hodl.drawdown",
            lower=sp.LineDefinition(
                label="hodl.drawdown",
                color=self.style.colors.hodl.rgba,
                width=self.style.line_width,
            ),
            color=self.style.colors.hodl_fill.rgba,  # fill_color
            opacity=0.1,
        )

        strategy_drawdown_fill_area = sp.ChannelDefinition(
            label="b.drawdown",
            lower=sp.LineDefinition(
                label="b.drawdown",
                color=self.style.colors.strategy.rgba,
                width=self.style.line_width,
            ),
            color=self.style.colors.strategy_fill.rgba,  # semi-transparent color,
            opacity=0.1,
        )

        capital_drawdown_fill_area = sp.ChannelDefinition(
            label="cptl.drawdown",
            lower=sp.LineDefinition(
                label="cptl.drawdown",
                color=self.style.colors.capital.rgba,
                width=self.style.line_width,
            ),
            color=self.style.colors.capital_fill.rgba,  # fill_color
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
                sp.LineDefinition(
                    label="hodl.value",
                    color=self.style.colors.hodl.rgba,
                    width=self.style.line_width
                ),
                sp.LineDefinition(
                    label="b.value",
                    color=self.style.colors.strategy.rgba,
                    width=self.style.line_width
                ),
                sp.LineDefinition(
                    label="cptl.b",
                    color=self.style.colors.capital.rgba,
                    width=self.style.line_width
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
            sub=(drawdown, portfolio,),
        )
