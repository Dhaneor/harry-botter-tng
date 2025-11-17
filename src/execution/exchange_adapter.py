#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExchangeAdapter Protocol — the unified interface for all exchanges.

This protocol defines the exact surface your bot will use, regardless of
which concrete exchange (Bybit, Binance, ...), or which backend (CCXT, mock).
"""

from __future__ import annotations

from typing import Protocol, Optional, List, Dict, Any
from datetime import datetime

from src.models.symbol import Asset, Market
from src.models.order import Order


class ExchangeAdapter(Protocol):
    """
    High-level, exchange-agnostic interface the bot requires.

    This interface abstracts CCXT and Bybit-specific behavior,
    and makes it easy to swap exchanges if needed.
    """

    # ---------------------------------------------------------
    # Public market data (no API key required)
    # ---------------------------------------------------------

    def get_market(self, symbol: str) -> Market:
        """Return the Market dataclass for a specific trading pair."""

    def get_all_markets(self) -> List[Market]:
        """Return all available markets (cached or freshly loaded)."""

    def fetch_ticker_price(self, symbol: str) -> float:
        """Return the last price or best bid/ask midpoint."""

    def fetch_asset_price(self, base: str, quote: str) -> float:
        """
        Convert any asset to a quote asset using the best available market.

        Example:
            fetch_asset_price("BTC", "USDT") -> 52000.0
        """

    # ---------------------------------------------------------
    # Account information (requires API key)
    # ---------------------------------------------------------

    def fetch_balance(self, asset: str) -> float:
        """Return free + used balance of a specific asset."""

    def fetch_all_balances(self) -> Dict[str, float]:
        """Return balances for all assets in the account."""

    def fetch_total_balance_in(self, quote: str) -> Optional[float]:
        """
        Return total account value expressed in a quote asset (e.g. USDT or BTC).
        Return None if the exchange/CCXT does not provide this directly.
        """

    # ---------------------------------------------------------
    # Order operations (private trading)
    # ---------------------------------------------------------

    def create_order(self, order: Order) -> Dict[str, Any]:
        """
        Place a real order at the exchange.
        Should return an exchange response containing at least:
            { "id": "...", "status": "...", ... }
        """

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order on the exchange."""

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch the current status of a specific order."""

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all open orders for a symbol or all markets."""

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def server_time(self) -> datetime:
        """Return the exchange server time for latency measurement."""

    def close(self) -> None:
        """Release any session resources."""