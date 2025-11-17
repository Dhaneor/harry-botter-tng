
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Optional

from src.models.order import Order
from src.execution.order_task import OrderTask


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


def make_order(**kwargs) -> Order:
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
# Lifecycle: valid order (no corrections needed)
# ============================================================

def test_ordertask_valid_order_flow():
    order = make_order()
    task = OrderTask(order=order)

    task.run_initial_validation()
    assert task.status == "ready"
    assert task.errors_initial == []
    assert task.final_order is order
    assert task.t_validated is not None
    assert task.t_ready is not None

    task.run_correction()
    assert task.corrected_order is None
    assert task.corrections == []

    task.run_final_validation()
    assert task.status == "ready"
    assert task.final_order is order
    assert task.fatal_errors() == []


# ============================================================
# Lifecycle: correctable order
# ============================================================

def test_ordertask_correctable_order_flow():
    m = DummyMarket()
    m.max_amount = 10.0

    order = make_order(market=m, amount=12.0)
    task = OrderTask(order=order)

    task.run_initial_validation()
    assert task.status == "validated"
    assert task.errors_initial != []
    assert task.final_order is None

    task.run_correction()
    assert task.corrected_order is not None
    assert task.corrected_order.amount == 10.0
    assert task.corrections != []
    assert task.t_corrected is not None

    task.run_final_validation()
    assert task.status == "ready"
    assert task.final_order.amount == 10.0
    assert task.errors_final == []


# ============================================================
# Lifecycle: fatal error (cannot be corrected)
# ============================================================

def test_ordertask_fatal_error_flow():
    m = DummyMarket()
    m.min_amount = 1.0

    order = make_order(market=m, amount=0.5)
    task = OrderTask(order=order)

    task.run_initial_validation()
    assert task.status == "validated"
    assert task.errors_initial != []

    task.run_correction()
    task.run_final_validation()

    assert task.status == "rejected"
    assert task.final_order is None
    assert task.errors_final != []
    assert task.fatal_errors() == task.errors_final


# ============================================================
# Submission + Fill + Fail
# ============================================================

def test_ordertask_submission_and_fill():
    order = make_order()
    task = OrderTask(order=order)

    task.final_order = order
    task.status = "ready"

    submit_resp = {"id": "abc123", "status": "open"}
    task.mark_submitted(submit_resp)

    assert task.status == "submitted"
    assert task.exchange_order_id == "abc123"
    assert task.exchange_response == submit_resp
    assert task.t_submitted is not None

    fill_resp = {"id": "abc123", "status": "closed"}
    task.mark_filled(fill_resp)

    assert task.status == "filled"
    assert task.exchange_order_id == "abc123"
    assert task.exchange_response == fill_resp
    assert task.t_filled is not None


def test_ordertask_mark_failed():
    order = make_order()
    task = OrderTask(order=order)

    task.mark_failed("network error")

    assert task.status == "failed"
    assert task.exchange_response == {"error": "network error"}
