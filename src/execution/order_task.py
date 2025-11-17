#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrderTask — Execution lifecycle object for handling an Order
from validation to correction to submission and final result.

This module uses a "dumb" OrderTask dataclass plus pure functions
that operate on it. The OrderTask holds mutable execution state,
while Order (in src.models.order) remains immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Literal, Any

from src.models.order import Order, validate_order, correct_order


OrderStatus = Literal[
    "created",
    "validated",
    "corrected",
    "ready",
    "submitted",
    "filled",
    "rejected",
    "failed",
]


@dataclass
class OrderTask:
    """
    Dumb state container for the lifecycle of an Order.

    All behavior (validation, correction, submission marking) lives
    in the pure functions defined below.
    """

    # Immutable input
    order: Order

    # Results of validation/correction stages
    corrected_order: Optional[Order] = None
    final_order: Optional[Order] = None

    # Order lifecycle status
    status: OrderStatus = "created"

    # Timestamps
    t_created: datetime = field(default_factory=datetime.now)
    t_validated: Optional[datetime] = None
    t_corrected: Optional[datetime] = None
    t_ready: Optional[datetime] = None
    t_submitted: Optional[datetime] = None
    t_filled: Optional[datetime] = None

    # Validation/correction logs
    errors_initial: List[str] = field(default_factory=list)
    errors_final: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)

    # Exchange response data
    exchange_order_id: Optional[str] = None
    exchange_response: Optional[Any] = None


# ----------------------------------------------------------------------
# Validation + Correction Pipeline (pure functions operating on OrderTask)
# ----------------------------------------------------------------------


def run_initial_validation(task: OrderTask) -> None:
    """
    Run the first validation pass on the original order.

    - Populates task.errors_initial
    - Sets t_validated
    - If no errors: sets final_order, status="ready", t_ready
    - Else: status="validated"
    """
    task.errors_initial = validate_order(task.order)
    task.t_validated = datetime.now()

    if not task.errors_initial:
        task.final_order = task.order
        task.status = "ready"
        task.t_ready = datetime.now()
    else:
        task.status = "validated"


def run_correction(task: OrderTask) -> None:
    """
    Run correctors based on initial errors (if any).

    - If there are no initial errors, does nothing.
    - Otherwise, calls correct_order() and updates:
        * corrected_order
        * corrections
        * t_corrected
        * status="corrected"
    """
    if not task.errors_initial:
        # Nothing to correct
        return

    corrected, logs = correct_order(task.order, task.errors_initial)
    task.corrected_order = corrected
    task.corrections = logs
    task.t_corrected = datetime.now()
    task.status = "corrected"


def run_final_validation(task: OrderTask) -> None:
    """
    Validate the corrected order (if present) or the original order.

    - Populates task.errors_final
    - If no errors: sets final_order, status="ready", t_ready
    - Else: status="rejected"
    """
    target = task.corrected_order or task.order
    task.errors_final = validate_order(target)

    if not task.errors_final:
        task.final_order = target
        task.status = "ready"
        task.t_ready = datetime.now()
    else:
        task.status = "rejected"


# ----------------------------------------------------------------------
# Submission + Execution (still pure functions mutating task state)
# ----------------------------------------------------------------------


def mark_submitted(task: OrderTask, exchange_response: dict) -> None:
    """
    Record that the order was submitted to the exchange.
    """
    task.exchange_response = exchange_response
    task.exchange_order_id = exchange_response.get("id")
    task.t_submitted = datetime.now()
    task.status = "submitted"


def mark_filled(task: OrderTask, fill_response: dict) -> None:
    """
    Mark the order as filled.
    """
    task.exchange_response = fill_response
    task.exchange_order_id = fill_response.get("id", task.exchange_order_id)
    task.t_filled = datetime.now()
    task.status = "filled"


def mark_failed(task: OrderTask, error_message: str) -> None:
    """
    Mark order as failed at submission level.
    """
    task.exchange_response = {"error": error_message}
    task.status = "failed"


# ----------------------------------------------------------------------
# Convenience predicates
# ----------------------------------------------------------------------


def is_ready(task: OrderTask) -> bool:
    return task.status == "ready"


def is_rejected(task: OrderTask) -> bool:
    return task.status == "rejected"


def is_submitted(task: OrderTask) -> bool:
    return task.status == "submitted"


def is_filled(task: OrderTask) -> bool:
    return task.status == "filled"


def fatal_errors(task: OrderTask) -> List[str]:
    """
    Errors after the second validation (fatal).
    """
    return task.errors_final