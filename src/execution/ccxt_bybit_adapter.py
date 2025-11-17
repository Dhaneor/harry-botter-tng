#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CcxtBybitExchangeAdapter — skeleton implementation.

Implements the ExchangeAdapter Protocol using CCXT's Bybit client.
All methods are stubs for now and will be filled step-by-step.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime

import ccxt

from src.execution.exchange_adapter import ExchangeAdapter
from src.models.symbol import Asset, Market
from src.models.order import Order


class CcxtBybitExchangeAdapter(ExchangeAdapter):
    """
    Concrete adapter using CCXT's unified Bybit interface.

    This class wraps CCXT in a clean, exchange-independent surface.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        load_markets: bool = True,
    ) -> None:
        # Create underlying CCXT exchange instance
        self.client = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        if testnet:
            # CCXT requires this explicit config for Bybit testnet
            self.client.options["defaultType"] = "linear"
            self.client.urls["api"] = self.client.urls["test"]

        self._markets: Dict[str, Market] = {}

        if load_markets:
            self._load_markets()

    # ---------------------------------------------------------
    # Internal helpers (not part of the public interface)
    # ---------------------------------------------------------

    def _load_markets(self) -> None:
        """Load and convert CCXT's raw markets to Market dataclasses."""
        # implementation will be added later
        pass

    def _get_ccxt_market(self, symbol: str) -> dict:
        """Return raw CCXT market dict."""
        # implementation will be added later
        raise NotImplementedError

    # ---------------------------------------------------------
    # Public market data
    # ---------------------------------------------------------

    def get_market(self, symbol: str) -> Market:
        pass

    def get_all_markets(self) -> List[Market]:
        pass

    def fetch_ticker_price(self, symbol: str) -> float:
        pass

    def fetch_asset_price(self, base: str, quote: str) -> float:
        pass

    # ---------------------------------------------------------
    # Account information
    # ---------------------------------------------------------

    def fetch_balance(self, asset: str) -> float:
        pass

    def fetch_all_balances(self) -> Dict[str, float]:
        pass

    def fetch_total_balance_in(self, quote: str) -> Optional[float]:
        pass

    # ---------------------------------------------------------
    # Order operations
    # ---------------------------------------------------------

    def create_order(self, order: Order) -> Dict[str, Any]:
        pass

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        pass

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        pass

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def server_time(self) -> datetime:
        pass

    def close(self) -> None:
        """Release CCXT resources."""
        if hasattr(self.client, "close"):
            self.client.close()