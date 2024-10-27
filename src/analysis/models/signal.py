#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a signal class for strategy and oracle classes.

Created on Sun Dec 11 19:08:20 2022

@author dhaneor
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd


@dataclass
class Signal:
    symbol: str
    interval: str
    data: Optional[pd.DataFrame]

    open_long: bool
    close_long: bool
    open_short: bool
    close_short: bool
    confidence: int

    stop_loss_long: Optional[Tuple[Tuple[float, float]]]
    stop_loss_short: Optional[Tuple[Tuple[float, float]]]
    take_profit_long: Optional[Tuple[Tuple[float, float]]]
    take_profit_short: Optional[Tuple[Tuple[float, float]]]
