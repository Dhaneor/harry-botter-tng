#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 15:05:23 2021

@author_ dhaneor
"""
import sys
import os

# ------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from src.exchange.kucoin_ import Kucoin  # noqa: F401, E402
from src.exchange.util.repositories import Repository  # noqa: F401, E402
from src.exchange.util.event_bus import (  # noqa: F401, E402
    EventBus,
    LoanRepaidEvent,
    OrderCreatedEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
)
from helpers.timeops import execution_time  # noqa: F401, E402
from broker.config import CREDENTIALS  # noqa: F401, E402


CLIENT = Kucoin(market="CROSS MARGIN", credentials=CREDENTIALS)


@execution_time
def check_latency():
    """
    Measures and prints the average roundtrip latency time for the Kucoin client.

    This function uses the Kucoin client to calculate the average latency
    of network requests and prints the result in milliseconds.

    Returns:
        None
    """
    res = CLIENT._get_average_latency()
    print("average roundtrip time (ms): ", res)


@execution_time
def get_repository():
    return Repository(client=CLIENT)


@execution_time
def check_repository(repo):
    for _ in range(10):
        acc = repo.account
        tic = repo.tickers
        sym = repo.symbols
        ord_ = repo.orders

    print(f"got account with {len(acc)} balances")
    print(f"got symbols list with {len(sym)} symbols")
    print(f"got tickers list with {len(tic)} tickers")
    print(f"got orders list with {len(ord_)} orders")


def test_async_repository():
    check_latency()
    repo = get_repository()
    check_repository(repo)


# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == "__main__":

    test_async_repository()
    # asyncio.run(amain())
