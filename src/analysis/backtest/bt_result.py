#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:56:23 2025

@author: dhaneor
"""
import numpy as np
import pandas as pd
from analysis import MarketData, POSITION_DTYPE, PORTFOLIO_DTYPE
from util.proj_types import Array_1D, Array_2D, Array_3D


class BackTestResult:
    def __init__(
            self, 
            positions: Array_2D, 
            portfolio: Array_2D,
            market_data: MarketData,
            signals: Array_2D
    ):
        self.positions = positions
        self.portfolio = portfolio
        self.market_data = market_data
        self.signals = signals

        self.symbols: list[str] = market_data.symbols

    def to_dataframe(self, symbol: str | None = None) -> pd.DataFrame:
        if symbol is not None:
            return self._build_df_symbol(symbol)
        else:
            return self._build_df_all()

    def _build_df_all(self) -> pd.DataFrame:
        ...

    def _build_df_symbol(self, symbol: str) -> pd.DataFrame:
        df = self.market_data[symbol]

        symbol_idx = self._get_symbol_index(symbol)
        df["signal"] = self.signals[:, symbol_idx]
        df["position"] = self.positions[:, symbol_idx]["position"]

        df["buy"] = self.positions[:, symbol_idx]["buy_qty"]
        df["buy_at"] = self.positions[:, symbol_idx]["buy_price"]
        df["sell"] = self.positions[:, symbol_idx]["sell_qty"]
        df["sell_at"] = self.positions[:, symbol_idx]["sell_price"]

        df["b.base"] = self.positions[:, symbol_idx]["qty"]
        df["b.quote"] = self.positions[:, symbol_idx]["quote_qty"]
        df["b.value"] = self.positions[:, symbol_idx]["equity"] # + df["b.quote"]

        return df

    def _get_symbol_index(self, symbol: str) -> int:
        return self.symbols.index(symbol)