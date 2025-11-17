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
    ERR_PRICE_TOO_LOW,
    ERR_PRICE_TOO_HIGH,
    ERR_AMOUNT_TOO_SMALL,
    ERR_AMOUNT_TOO_LARGE,
    ERR_MIN_NOTIONAL,
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
    correct_order,
)


# ------------------------------------------------------------
# Minimal Market stub
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
# Helper
# ------------------------------------------------------------

def make_order(**kwargs):
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
# Structural validators
# ============================================================

def test_validate_market_attached_ok():
    order = make_order()
    assert validate_market_attached(order, order.market) is None


def test_validate_market_attached_missing():
    order = make_order(market=None)
    assert validate_market_attached(order, None) == ERR_MARKET_NOT_ATTACHED


def test_validate_side_ok():
    assert validate_side(make_order(side="buy"), DummyMarket()) is None
    assert validate_side(make_order(side="sell"), DummyMarket()) is None


def test_validate_side_invalid():
    assert validate_side(make_order(side="xxx"), DummyMarket()) == ERR_INVALID_SIDE


def test_validate_order_type_ok():
    valid = [
        "market","limit","stop_market","stop_limit","stop_loss","stop_loss_limit",
        "take_profit","take_profit_market","take_profit_limit",
        "trailing_stop","trailing_stop_market",
    ]
    for t in valid:
        assert validate_order_type(make_order(type=t), DummyMarket()) is None


def test_validate_order_type_invalid():
    assert validate_order_type(make_order(type="iceberg"), DummyMarket()) == ERR_UNSUPPORTED_ORDER_TYPE


def test_validate_price_requirement_market_ok():
    order = make_order(type="market", price=None)
    assert validate_price_requirement(order, DummyMarket()) is None


def test_validate_price_requirement_market_disallowed():
    order = make_order(type="market", price=100)
    assert validate_price_requirement(order, DummyMarket()) == ERR_PRICE_NOT_ALLOWED


def test_validate_price_requirement_limit_missing():
    order = make_order(type="limit", price=None)
    assert validate_price_requirement(order, DummyMarket()) == ERR_PRICE_REQUIRED


def test_validate_price_requirement_limit_ok():
    order = make_order(type="limit", price=100)
    assert validate_price_requirement(order, DummyMarket()) is None


def test_validate_stop_price_required_missing():
    order = make_order(type="stop_market", stop_price=None)
    assert validate_stop_price_requirement(order, DummyMarket()) == ERR_STOP_PRICE_REQUIRED


def test_validate_stop_price_not_allowed():
    order = make_order(type="limit", stop_price=123)
    assert validate_stop_price_requirement(order, DummyMarket()) == ERR_STOP_PRICE_NOT_ALLOWED


def test_validate_stop_price_ok():
    order = make_order(type="stop_market", stop_price=123)
    assert validate_stop_price_requirement(order, DummyMarket()) is None


def test_validate_amount_positive_ok():
    assert validate_amount_positive(make_order(amount=0.1), DummyMarket()) is None


def test_validate_amount_positive_invalid():
    assert validate_amount_positive(make_order(amount=0), DummyMarket()) == ERR_AMOUNT_NOT_POSITIVE
    assert validate_amount_positive(make_order(amount=-1), DummyMarket()) == ERR_AMOUNT_NOT_POSITIVE


# ============================================================
# Precision validators
# ============================================================

def test_price_precision_ok():
    m = DummyMarket()
    m.price_precision = 2
    order = make_order(price=100.12)
    assert validate_price_precision(order, m) is None


def test_price_precision_invalid():
    m = DummyMarket()
    m.price_precision = 2
    order = make_order(price=100.123)
    assert validate_price_precision(order, m) == ERR_PRICE_PRECISION


def test_amount_precision_ok():
    m = DummyMarket()
    m.amount_precision = 3
    order = make_order(amount=0.123)
    assert validate_amount_precision(order, m) is None


def test_amount_precision_invalid():
    m = DummyMarket()
    m.amount_precision = 3
    order = make_order(amount=0.12345)
    assert validate_amount_precision(order, m) == ERR_AMOUNT_PRECISION


# ============================================================
# Min / Max validators
# ============================================================

def test_price_min_invalid():
    m = DummyMarket()
    m.min_price = 100
    order = make_order(price=99)
    assert validate_price_min(order, m) == ERR_PRICE_TOO_LOW


def test_price_min_ok():
    m = DummyMarket()
    m.min_price = 100
    order = make_order(price=100)
    assert validate_price_min(order, m) is None


def test_price_max_invalid():
    m = DummyMarket()
    m.max_price = 200
    order = make_order(price=201)
    assert validate_price_max(order, m) == ERR_PRICE_TOO_HIGH


def test_price_max_ok():
    m = DummyMarket()
    m.max_price = 200
    order = make_order(price=200)
    assert validate_price_max(order, m) is None


def test_amount_min_invalid():
    m = DummyMarket()
    m.min_amount = 1
    order = make_order(amount=0.5)
    assert validate_amount_min(order, m) == ERR_AMOUNT_TOO_SMALL


def test_amount_min_ok():
    m = DummyMarket()
    m.min_amount = 1
    order = make_order(amount=1)
    assert validate_amount_min(order, m) is None


def test_amount_max_invalid():
    m = DummyMarket()
    m.max_amount = 10
    order = make_order(amount=11)
    assert validate_amount_max(order, m) == ERR_AMOUNT_TOO_LARGE


def test_amount_max_ok():
    m = DummyMarket()
    m.max_amount = 10
    order = make_order(amount=5)
    assert validate_amount_max(order, m) is None


# ============================================================
# Min cost validator
# ============================================================

def test_min_cost_invalid():
    m = DummyMarket()
    m.min_cost = 50
    order = make_order(price=10, amount=4)
    assert validate_min_cost(order, m) == ERR_MIN_NOTIONAL


def test_min_cost_ok():
    m = DummyMarket()
    m.min_cost = 50
    order = make_order(price=10, amount=5)
    assert validate_min_cost(order, m) is None


# ============================================================
# validate_order orchestrator
# ============================================================

def test_validate_order_ok():
    m = DummyMarket()
    order = make_order()
    assert validate_order(order, m) == []


def test_validate_order_multiple_errors():
    m = DummyMarket()
    m.price_precision = 2
    m.min_amount = 1

    order = make_order(price=100.123, amount=0.5)

    errors = validate_order(order, m)
    assert ERR_PRICE_PRECISION in errors
    assert ERR_AMOUNT_TOO_SMALL in errors


# ============================================================
# Corrector tests
# ============================================================

def test_correct_order_price_precision():
    m = DummyMarket()
    m.price_precision = 2

    order = make_order(price=100.123)
    err1 = validate_order(order, m)
    corrected, logs = correct_order(order, err1, m)

    assert corrected.price == 100.12
    assert any("price_precision" in msg for msg in logs)
    assert validate_order(corrected, m) == []


def test_correct_order_amount_max():
    m = DummyMarket()
    m.max_amount = 10

    order = make_order(amount=12)
    err1 = validate_order(order, m)
    corrected, logs = correct_order(order, err1, m)

    assert corrected.amount == 10
    assert any("amount_max" in msg for msg in logs)
    assert validate_order(corrected, m) == []


def test_correct_order_market_price_removed():
    m = DummyMarket()
    order = make_order(type="market", price=100)

    err1 = validate_order(order, m)
    corrected, logs = correct_order(order, err1, m)

    assert corrected.price is None
    assert any("price_not_allowed" in msg for msg in logs)
    assert validate_order(corrected, m) == []
