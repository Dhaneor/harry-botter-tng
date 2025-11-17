#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Optional

from src.models.order import Order
from src.execution.order_task import (
    OrderTask,
    run_initial_validation,
    run_correction,
    run_final_validation,
    mark_submitted,
    mark_filled,
    mark_failed,
    is_ready,
    is_rejected,
    is_submitted,
    is_filled,
    fatal_errors,
)


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

    # Initial validation should mark it ready directly
    run_initial_validation(task)

    assert task.status == "ready"
    assert task.errors_initial == []
    assert task.final_order is order
    assert task.t_validated is not None
    assert task.t_ready is not None
    assert is_ready(task) is True
    assert is_rejected(task) is False

    # Running correction should be a no-op
    run_correction(task)
    assert task.corrected_order is None
    assert task.corrections == []

    # Final validation should keep it ready
    run_final_validation(task)
    assert task.status == "ready"
    assert task.final_order is order
    assert fatal_errors(task) == []


# ============================================================
# Lifecycle: correctable order (too large amount, then corrected)
# ============================================================

def test_ordertask_correctable_order_flow():
    m = DummyMarket()
    m.max_amount = 10.0

    order = make_order(market=m, amount=12.0)
    task = OrderTask(order=order)

    # First pass: should detect errors and not be ready
    run_initial_validation(task)
    assert task.status == "validated"
    assert task.errors_initial != []
    assert task.final_order is None

    # Correction should reduce amount and log corrections
    run_correction(task)
    assert task.corrected_order is not None
    assert task.corrected_order.amount == 10.0
    assert task.corrections != []
    assert task.t_corrected is not None

    # Final validation should now pass and mark as ready
    run_final_validation(task)
    assert task.status == "ready"
    assert task.final_order is not None
    assert task.final_order.amount == 10.0
    assert task.errors_final == []
    assert is_ready(task) is True
    assert is_rejected(task) is False


# ============================================================
# Lifecycle: fatal error (amount too small cannot be corrected)
# ============================================================

def test_ordertask_fatal_error_flow():
    m = DummyMarket()
    m.min_amount = 1.0

    order = make_order(market=m, amount=0.5)
    task = OrderTask(order=order)

    run_initial_validation(task)
    assert task.status == "validated"
    assert task.errors_initial != []

    run_correction(task)
    # No corrector exists for "amount too small", so final validation
    # should still fail and reject the order.
    run_final_validation(task)

    assert task.status == "rejected"
    assert task.final_order is None
    assert task.errors_final != []
    assert is_ready(task) is False
    assert is_rejected(task) is True
    assert len(fatal_errors(task)) == len(task.errors_final)


# ============================================================
# Submission / execution flags
# ============================================================

def test_ordertask_submission_and_fill():
    order = make_order()
    task = OrderTask(order=order)

    # Pretend it's already validated and ready
    task.final_order = order
    task.status = "ready"

    # Mark submitted
    submit_resp = {"id": "abc123", "status": "open"}
    mark_submitted(task, submit_resp)

    assert task.status == "submitted"
    assert task.exchange_order_id == "abc123"
    assert task.exchange_response == submit_resp
    assert task.t_submitted is not None
    assert is_submitted(task) is True

    # Mark filled
    fill_resp = {"id": "abc123", "status": "closed"}
    mark_filled(task, fill_resp)

    assert task.status == "filled"
    assert task.exchange_order_id == "abc123"
    assert task.exchange_response == fill_resp
    assert task.t_filled is not None
    assert is_filled(task) is True


def test_ordertask_mark_failed():
    order = make_order()
    task = OrderTask(order=order)

    mark_failed(task, "network error")
    assert task.status == "failed"
    assert task.exchange_response == {"error": "network error"}
    assert is_filled(task) is False
    assert is_submitted(task) is False
    assert is_ready(task) is False