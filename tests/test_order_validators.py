#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from datetime import datetime
from typing import Optional

from src.models.order import (
    Order,
    ERR_MARKET_NOT_ATTACHED,
    ERR_INVALID_SIDE,
    ERR_UNSUPPORTED_ORDER_TYPE,
    ERR_PRICE_REQUIRED,
    ERR_PRICE_NOT_ALLOWED,
    ERR_STOP_PRICE_REQUIRED,
    ERR_STOP_PRICE_NOT_ALLOWED,
    ERR_AMOUNT_NOT_POSITIVE,
    ERR_PRICE_PRECISION,
    ERR_AMOUNT_PRECISION,
    ERR_MIN_NOTIONAL,
    ERR_PRICE_TOO_LOW,
    ERR_PRICE_TOO_HIGH,
    ERR_AMOUNT_TOO_SMALL,
    ERR_AMOUNT_TOO_LARGE,
    validate_market_attached,
    validate_side,
    validate_order_type,
    validate_price_requirement,
    validate_stop_price_requirement,
    validate_amount_positive,
    validate_price_precision,
    validate_amount_precision,
    validate_price_min,
    validate_price_max,
    validate_amount_min,
    validate_amount_max,
    validate_min_cost,
    validate_order,
)


# ------------------------------------------------------------
# Minimal Market stub — we do not test Market here.
# ------------------------------------------------------------

class DummyMarket:
    def __init__(self):
        self.symbol = "BTC/USDT"
        self.price_precision: Optional[int] = None
        self.amount_precision: Optional[int] = None
        self.min_price: Optional[float] = None
        self.max_price: Optional[float] = None
        self.min_amount: Optional[float] = None
        self.max_amount: Optional[float] = None
        self.min_cost: Optional[float] = None


# ------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------

def make_order(**kwargs):
    """Create a valid baseline order, allowing overrides via kwargs."""
    defaults = {
        "market": DummyMarket(),
        "side": "buy",
        "type": "limit",
        "amount": 1.0,
        "price": 100.0,
        "stop_price": None,
        "client_id": None,
        "created_at": datetime.now(),
        "extra": None,
    }
    defaults.update(kwargs)
    return Order(**defaults)


# ============================================================
# validate_market_attached
# ============================================================

def test_validate_market_attached_ok():
    order = make_order()
    err = validate_market_attached(order, order.market)
    assert err is None


def test_validate_market_attached_missing_market():
    order = make_order(market=None)
    err = validate_market_attached(order, None)
    assert err == ERR_MARKET_NOT_ATTACHED


# ============================================================
# validate_side
# ============================================================

def test_validate_side_ok():
    assert validate_side(make_order(side="buy"), DummyMarket()) is None
    assert validate_side(make_order(side="sell"), DummyMarket()) is None


def test_validate_side_invalid():
    err = validate_side(make_order(side="invalid"), DummyMarket())
    assert err == ERR_INVALID_SIDE


# ============================================================
# validate_order_type
# ============================================================

def test_validate_order_type_ok():
    valid_types = [
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
    for t in valid_types:
        assert validate_order_type(make_order(type=t), DummyMarket()) is None


def test_validate_order_type_invalid():
    err = validate_order_type(make_order(type="iceberg"), DummyMarket())
    assert err == ERR_UNSUPPORTED_ORDER_TYPE


# ============================================================
# validate_price_requirement
# ============================================================

def test_validate_price_requirement_market_order_ok():
    order = make_order(type="market", price=None)
    err = validate_price_requirement(order, DummyMarket())
    assert err is None


def test_validate_price_requirement_market_order_price_not_allowed():
    order = make_order(type="market", price=100.0)
    err = validate_price_requirement(order, DummyMarket())
    assert err == ERR_PRICE_NOT_ALLOWED


def test_validate_price_requirement_limit_requires_price():
    order = make_order(type="limit", price=None)
    err = validate_price_requirement(order, DummyMarket())
    assert err == ERR_PRICE_REQUIRED


def test_validate_price_requirement_limit_ok():
    order = make_order(type="limit", price=100.0)
    err = validate_price_requirement(order, DummyMarket())
    assert err is None


# ============================================================
# validate_stop_price_requirement
# ============================================================

def test_validate_stop_price_requirement_stop_market_ok():
    order = make_order(type="stop_market", stop_price=99.5)
    err = validate_stop_price_requirement(order, DummyMarket())
    assert err is None


def test_validate_stop_price_requirement_stop_market_missing():
    order = make_order(type="stop_market", stop_price=None)
    err = validate_stop_price_requirement(order, DummyMarket())
    assert err == ERR_STOP_PRICE_REQUIRED


def test_validate_stop_price_requirement_non_stop_must_not_have_stop_price():
    order = make_order(type="limit", stop_price=100)
    err = validate_stop_price_requirement(order, DummyMarket())
    assert err == ERR_STOP_PRICE_NOT_ALLOWED


def test_validate_stop_price_requirement_non_stop_no_stop_price_ok():
    order = make_order(type="limit", stop_price=None)
    err = validate_stop_price_requirement(order, DummyMarket())
    assert err is None


# ============================================================
# validate_amount_positive
# ============================================================

def test_validate_amount_positive_ok():
    order = make_order(amount=0.001)
    err = validate_amount_positive(order, DummyMarket())
    assert err is None


def test_validate_amount_positive_zero():
    order = make_order(amount=0.0)
    err = validate_amount_positive(order, DummyMarket())
    assert err == ERR_AMOUNT_NOT_POSITIVE


def test_validate_amount_positive_negative():
    order = make_order(amount=-1.0)
    err = validate_amount_positive(order, DummyMarket())
    assert err == ERR_AMOUNT_NOT_POSITIVE


# ============================================================
# validate_price_precision
# ============================================================

def test_validate_price_precision_ok():
    m = DummyMarket()
    m.price_precision = 2
    order = make_order(type="limit", price=100.12)
    err = validate_price_precision(order, m)
    assert err is None


def test_validate_price_precision_too_many_decimals():
    m = DummyMarket()
    m.price_precision = 2
    order = make_order(type="limit", price=100.123)
    err = validate_price_precision(order, m)
    assert err == ERR_PRICE_PRECISION


def test_validate_price_precision_ignored_for_market_orders():
    m = DummyMarket()
    m.price_precision = 2
    order = make_order(type="market", price=None)
    err = validate_price_precision(order, m)
    assert err is None


# ============================================================
# validate_amount_precision
# ============================================================

def test_validate_amount_precision_ok():
    m = DummyMarket()
    m.amount_precision = 3
    order = make_order(amount=0.123)
    err = validate_amount_precision(order, m)
    assert err is None


def test_validate_amount_precision_too_many_decimals():
    m = DummyMarket()
    m.amount_precision = 3
    order = make_order(amount=0.12345)
    err = validate_amount_precision(order, m)
    assert err == ERR_AMOUNT_PRECISION


# ============================================================
# validate_price_min / validate_price_max
# ============================================================

def test_validate_price_min_below():
    m = DummyMarket()
    m.min_price = 100.0
    order = make_order(price=99.9)
    err = validate_price_min(order, m)
    assert err == ERR_PRICE_TOO_LOW


def test_validate_price_min_ok():
    m = DummyMarket()
    m.min_price = 100.0
    order = make_order(price=100.0)
    err = validate_price_min(order, m)
    assert err is None


def test_validate_price_max_above():
    m = DummyMarket()
    m.max_price = 200.0
    order = make_order(price=200.1)
    err = validate_price_max(order, m)
    assert err == ERR_PRICE_TOO_HIGH


def test_validate_price_max_ok():
    m = DummyMarket()
    m.max_price = 200.0
    order = make_order(price=200.0)
    err = validate_price_max(order, m)
    assert err is None


# ============================================================
# validate_amount_min / validate_amount_max
# ============================================================

def test_validate_amount_min_below():
    m = DummyMarket()
    m.min_amount = 1.0
    order = make_order(amount=0.5)
    err = validate_amount_min(order, m)
    assert err == ERR_AMOUNT_TOO_SMALL


def test_validate_amount_min_ok():
    m = DummyMarket()
    m.min_amount = 1.0
    order = make_order(amount=1.0)
    err = validate_amount_min(order, m)
    assert err is None


def test_validate_amount_max_above():
    m = DummyMarket()
    m.max_amount = 10.0
    order = make_order(amount=11.0)
    err = validate_amount_max(order, m)
    assert err == ERR_AMOUNT_TOO_LARGE


def test_validate_amount_max_ok():
    m = DummyMarket()
    m.max_amount = 10.0
    order = make_order(amount=5.0)
    err = validate_amount_max(order, m)
    assert err is None


# ============================================================
# validate_min_cost
# ============================================================

def test_validate_min_cost_below_min():
    m = DummyMarket()
    m.min_cost = 50.0
    order = make_order(price=10.0, amount=4.0)  # cost = 40
    err = validate_min_cost(order, m)
    assert err == ERR_MIN_NOTIONAL


def test_validate_min_cost_above_min():
    m = DummyMarket()
    m.min_cost = 50.0
    order = make_order(price=10.0, amount=5.0)  # cost = 50
    err = validate_min_cost(order, m)
    assert err is None


# ============================================================
# validate_order orchestrator
# ============================================================

def test_validate_order_all_ok():
    m = DummyMarket()
    order = make_order(price=150.0, amount=1.0)
    errors = validate_order(order, m)
    assert errors == []


def test_validate_order_multiple_errors():
    m = DummyMarket()
    m.price_precision = 2
    m.amount_precision = 2
    m.min_amount = 1.0

    # too many decimals AND too small amount
    order = make_order(price=100.123, amount=0.5)

    errors = validate_order(order, m)

    assert ERR_PRICE_PRECISION in errors
    assert ERR_AMOUNT_TOO_SMALL in errors