#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 19:40:06 2025

@author: dhaneor
"""


from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, Dict, Any

from .symbol import Market

# ============================================================
# Order Validation Error Codes
# ============================================================

ERR_MARKET_NOT_ATTACHED = "market_not_attached"
ERR_INVALID_SIDE = "invalid_side"
ERR_UNSUPPORTED_ORDER_TYPE = "unsupported_order_type"
ERR_PRICE_REQUIRED = "price_required"
ERR_PRICE_NOT_ALLOWED = "price_not_allowed"
ERR_STOP_PRICE_REQUIRED = "stop_price_required"
ERR_STOP_PRICE_NOT_ALLOWED = "stop_price_not_allowed"
ERR_AMOUNT_NOT_POSITIVE = "amount_not_positive"
ERR_PRICE_PRECISION = "price_precision"
ERR_AMOUNT_PRECISION = "amount_precision"
ERR_PRICE_LIMITS = "price_limits"
ERR_AMOUNT_LIMITS = "amount_limits"
ERR_MIN_NOTIONAL = "min_notional"
ERR_PRICE_TOO_LOW = "price_too_low"
ERR_PRICE_TOO_HIGH = "price_too_high"
ERR_AMOUNT_TOO_SMALL = "amount_too_small"
ERR_AMOUNT_TOO_LARGE = "amount_too_large"


# ============================================================
# Basic Structural Validators
# ============================================================

def validate_market_attached(order, market):
    if order.market is None or market is None:
        return ERR_MARKET_NOT_ATTACHED
    return None


def validate_side(order, market):
    if order.side not in ("buy", "sell"):
        return ERR_INVALID_SIDE
    return None


def validate_order_type(order, market):
    VALID_TYPES = {
        "market",
        "limit",
        "stop_market",
        "stop_limit",
        "stop_loss",
        "stop_loss_limit",
        "take_profit",
        "take_profit_market",
        "take_profit_limit",
        "trailing_stop",
        "trailing_stop_market",
    }
    if order.type not in VALID_TYPES:
        return ERR_UNSUPPORTED_ORDER_TYPE
    return None


def validate_price_requirement(order, market):
    if order.type == "market":
        if order.price is not None:
            return ERR_PRICE_NOT_ALLOWED
        return None

    if order.type in ("limit", "stop_limit", "take_profit_limit"):
        if order.price is None:
            return ERR_PRICE_REQUIRED
        return None

    return None


def validate_stop_price_requirement(order, market):
    STOP_TYPES = {
        "stop_market",
        "stop_limit",
        "stop_loss",
        "stop_loss_limit",
        "take_profit",
        "take_profit_market",
        "take_profit_limit",
        "trailing_stop",
        "trailing_stop_market",
    }

    if order.type in STOP_TYPES:
        if order.stop_price is None:
            return ERR_STOP_PRICE_REQUIRED
        return None

    if order.stop_price is not None:
        return ERR_STOP_PRICE_NOT_ALLOWED

    return None


def validate_amount_positive(order, market):
    if order.amount <= 0:
        return ERR_AMOUNT_NOT_POSITIVE
    return None


# ============================================================
# Precision & Limit Helpers / Validators
# ============================================================

def fits_precision(value: float, precision: int) -> bool:
    """
    Return True if `value` can be represented with at most `precision`
    decimal places.
    """
    if precision is None:
        return True
    factor = 10 ** precision
    rounded = round(value * factor) / factor
    return rounded == value


def validate_price_precision(order, market):
    # Skip if price is not relevant for this order
    if order.price is None:
        return None
    prec = getattr(market, "price_precision", None)
    if prec is None:
        return None
    if not fits_precision(order.price, prec):
        return ERR_PRICE_PRECISION
    return None


def validate_amount_precision(order, market):
    prec = getattr(market, "amount_precision", None)
    if prec is None:
        return None
    if not fits_precision(order.amount, prec):
        return ERR_AMOUNT_PRECISION
    return None


def validate_price_min(order, market):
    if order.price is None:
        return None
    min_price = getattr(market, "min_price", None)
    if min_price is not None and order.price < min_price:
        return ERR_PRICE_TOO_LOW
    return None


def validate_price_max(order, market):
    if order.price is None:
        return None
    max_price = getattr(market, "max_price", None)
    if max_price is not None and order.price > max_price:
        return ERR_PRICE_TOO_HIGH
    return None


def validate_amount_min(order, market):
    min_amount = getattr(market, "min_amount", None)
    if min_amount is not None and order.amount < min_amount:
        return ERR_AMOUNT_TOO_SMALL
    return None


def validate_amount_max(order, market):
    max_amount = getattr(market, "max_amount", None)
    if max_amount is not None and order.amount > max_amount:
        return ERR_AMOUNT_TOO_LARGE
    return None


def validate_min_cost(order, market):
    """
    Validate that order notional (price * amount) is above the exchange's
    minimum notional, if defined.
    """
    min_cost = getattr(market, "min_cost", None)
    if min_cost is None:
        return None
    if order.price is None:
        return None

    cost = order.price * order.amount
    if cost < min_cost:
        return ERR_MIN_NOTIONAL
    return None


# ============================================================
# Order Validation Orchestrator
# ============================================================

# Ordered list of all validators
ALL_VALIDATORS = [
    # Structural checks
    validate_market_attached,
    validate_side,
    validate_order_type,
    validate_price_requirement,
    validate_stop_price_requirement,
    validate_amount_positive,

    # Precision / limit checks
    validate_price_precision,
    validate_amount_precision,
    validate_price_min,
    validate_price_max,
    validate_amount_min,
    validate_amount_max,
    validate_min_cost,
]


def validate_order(order, market=None):
    """
    Run all registered validators against an order.

    Returns:
        list[str] - a list of error codes, empty if valid.
    """
    if market is None:
        market = order.market

    errors = []
    for validator in ALL_VALIDATORS:
        err = validator(order, market)
        if err:
            errors.append(err)

    return errors


# ============================================================
# Order class
# ============================================================

OrderType = Literal[
    "market",
    "limit",
    "stop_market",
    "stop_limit",
    "stop_loss",
    "stop_loss_limit",
    "take_profit",
    "take_profit_market",
    "take_profit_limit",
    "trailing_stop",
    "trailing_stop_market",
]

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Order:
    """Represents an immutable trading instruction."""

    # -------------------------
    # Core order intent
    # -------------------------
    market: Market
    side: Side
    type: OrderType

    amount: float
    price: Optional[float]      # None for market or stop_market
    stop_price: Optional[float] # required for stop/trigger orders

    # -------------------------
    # Metadata
    # -------------------------
    client_id: Optional[str]    # clientOrderId (optional)
    created_at: datetime

    # Raw info (optional), useful for debugging or reconstructing orders
    extra: Optional[Dict[str, Any]] = None