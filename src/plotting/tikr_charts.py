#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
logger = logging.getLogger(f"main.{__name__}")


class TikrChartBase:
    def __init__(self, df, title):
        self.data_source = df
        self.title = title
        self.chart_data = None

    def draw(self):
        logger.info(f"Drawing chart for {self.title}")


class TikrChart(TikrChartBase):
    def __init__(self, df, title):
        super().__init__(df, title)
        # self.chart_data = self._generate_chart_data()