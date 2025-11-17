#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrderTask — Execution lifecycle object for handling an Order
from validation to correction to submission and final result.
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
    OrderTask tracks the complete lifecycle of an order:
    - structural validation
    - optional correction
    - re-validation
    - submission attempt
    - final exchange result

    NOTE: This class is intentionally MUTABLE.
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

    # ------------------------------------------------------------------
    # Validation + Correction Pipeline
    # ------------------------------------------------------------------

    def run_initial_validation(self) -> None:
        """Run the first validation pass on the original order."""
        self.errors_initial = validate_order(self.order)
        self.t_validated = datetime.now()

        if not self.errors_initial:
            # No issues at all: the order is already ready
            self.final_order = self.order
            self.status = "ready"
            self.t_ready = datetime.now()
        else:
            self.status = "validated"

    def run_correction(self) -> None:
        """Run correctors based on initial errors (if any)."""
        if not self.errors_initial:
            # Nothing to correct
            return

        corrected, logs = correct_order(self.order, self.errors_initial)
        self.corrected_order = corrected
        self.corrections = logs
        self.t_corrected = datetime.now()
        self.status = "corrected"

    def run_final_validation(self) -> None:
        """
        Validate the corrected order (if present) or the original order.

        If errors remain, they are considered fatal and the task is rejected.
        """
        target = self.corrected_order or self.order
        self.errors_final = validate_order(target)

        if not self.errors_final:
            self.final_order = target
            self.status = "ready"
            self.t_ready = datetime.now()
        else:
            self.status = "rejected"

    # ------------------------------------------------------------------
    # Submission + Execution
    # ------------------------------------------------------------------

    def mark_submitted(self, exchange_response: dict) -> None:
        """Record that the order was submitted to the exchange."""
        self.exchange_response = exchange_response
        self.exchange_order_id = exchange_response.get("id")
        self.t_submitted = datetime.now()
        self.status = "submitted"

    def mark_filled(self, fill_response: dict) -> None:
        """Mark the order as filled."""
        self.exchange_response = fill_response
        self.exchange_order_id = fill_response.get("id", self.exchange_order_id)
        self.t_filled = datetime.now()
        self.status = "filled"

    def mark_failed(self, error_message: str) -> None:
        """Mark order as failed at submission level."""
        self.exchange_response = {"error": error_message}
        self.status = "failed"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.status == "ready"

    def is_rejected(self) -> bool:
        return self.status == "rejected"

    def is_submitted(self) -> bool:
        return self.status == "submitted"

    def is_filled(self) -> bool:
        return self.status == "filled"

    def fatal_errors(self) -> List[str]:
        """Errors after second validation (fatal)."""
        return self.errors_final