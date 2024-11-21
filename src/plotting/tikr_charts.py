#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
logger = logging.getLogger(f"main.{__name__}")


class TikrChartBase:
    def __init__(self, data_source, ticker):
        self.data_source = data_source
        self.ticker = ticker
        self.chart_data = None

    def draw(self):
        ...