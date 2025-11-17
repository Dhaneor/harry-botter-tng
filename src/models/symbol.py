#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:00:06 2025

@author: dhaneor
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


# -------------------------
# Asset Dataclass
# -------------------------

@dataclass
class Asset:
    id: str                     # exchange-specific id (e.g. "BTC")
    code: str                   # unified CCXT code (e.g. "BTC")
    name: Optional[str]         # human-readable name (e.g. "Bitcoin")

    precision: Optional[int]    # number of decimal places allowed
    active: Optional[bool]
    withdraw: Optional[bool]
    deposit: Optional[bool]

    fee: Optional[float]        # withdrawal fee

    limits_withdraw: Optional[Dict[str, Optional[float]]]   # {min, max}
    limits_deposit: Optional[Dict[str, Optional[float]]]    # {min, max}

    # Computed flags
    is_stablecoin: bool
    is_fiat: bool
    is_crypto: bool

    # Raw exchange payload
    info: Dict[str, Any]


# -------------------------
# Market Dataclass
# -------------------------

@dataclass
class Market:
    symbol: str
    id: str
    type: Optional[str]            # "spot", "swap", etc.

    base: Optional[Asset]
    quote: Optional[Asset]
    settle: Optional[Asset]

    contract: Optional[bool]
    linear: Optional[bool]
    inverse: Optional[bool]
    contract_size: Optional[float]

    maker: Optional[float]
    taker: Optional[float]

    price_precision: Optional[int]
    amount_precision: Optional[int]
    cost_precision: Optional[int]

    min_amount: Optional[float]
    max_amount: Optional[float]
    min_price: Optional[float]
    max_price: Optional[float]
    min_cost: Optional[float]
    max_cost: Optional[float]

    # Convenience flags
    is_spot: bool
    is_linear_future: bool
    is_inverse_future: bool
    is_perpetual: bool

    # Raw exchange payload
    info: Dict[str, Any]


# -------------------------
# Conversion Helpers
# -------------------------

_STABLECOINS = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "BUSD", "PYUSD"}
_FIAT = {"EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}


def make_asset(data: Dict[str, Any]) -> Asset:
    """Convert a CCXT currency dict into an Asset instance."""
    
    code = data.get("code")
    
    is_fiat = code in _FIAT
    is_stable = code in _STABLECOINS
    is_crypto = not (is_fiat or is_stable)

    return Asset(
        id=data.get("id"),
        code=code,
        name=data.get("name"),

        precision=data.get("precision"),
        active=data.get("active"),
        withdraw=data.get("withdraw"),
        deposit=data.get("deposit"),

        fee=data.get("fee"),

        limits_withdraw=(data.get("limits", {}).get("withdraw")
                         if isinstance(data.get("limits"), dict) else None),
        limits_deposit=(data.get("limits", {}).get("deposit")
                        if isinstance(data.get("limits"), dict) else None),

        is_stablecoin=is_stable,
        is_fiat=is_fiat,
        is_crypto=is_crypto,

        info=data,
    )


def make_market(
    data: Dict[str, Any],
    assets: Dict[str, Asset]
) -> Market:
    """Convert a CCXT market dict into a Market instance.
       `assets` is a dict: {code -> Asset}
    """

    precision = data.get("precision", {}) or {}
    limits = data.get("limits", {}) or {}

    base = assets.get(data.get("base"))
    quote = assets.get(data.get("quote"))
    settle = assets.get(data.get("settle"))

    contract = data.get("contract")
    linear = data.get("linear")
    inverse = data.get("inverse")

    # Convenience flags
    type_ = data.get("type")
    is_spot = type_ == "spot"
    is_linear_future = bool(contract and linear)
    is_inverse_future = bool(contract and inverse)
    is_perpetual = bool(contract and not data.get("expiry"))

    return Market(
        symbol=data.get("symbol"),
        id=data.get("id"),
        type=type_,

        base=base,
        quote=quote,
        settle=settle,

        contract=contract,
        linear=linear,
        inverse=inverse,
        contract_size=data.get("contractSize"),

        maker=data.get("maker"),
        taker=data.get("taker"),

        price_precision=precision.get("price"),
        amount_precision=precision.get("amount"),
        cost_precision=precision.get("cost"),

        min_amount=limits.get("amount", {}).get("min") if "amount" in limits else None,
        max_amount=limits.get("amount", {}).get("max") if "amount" in limits else None,
        min_price=limits.get("price", {}).get("min") if "price" in limits else None,
        max_price=limits.get("price", {}).get("max") if "price" in limits else None,
        min_cost=limits.get("cost", {}).get("min") if "cost" in limits else None,
        max_cost=limits.get("cost", {}).get("max") if "cost" in limits else None,

        is_spot=is_spot,
        is_linear_future=is_linear_future,
        is_inverse_future=is_inverse_future,
        is_perpetual=is_perpetual,

        info=data,
    )