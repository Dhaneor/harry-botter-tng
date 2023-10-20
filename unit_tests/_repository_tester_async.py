#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 15:05:23 2021

@author_ dhaneor
"""
import sys
import os
import time
import asyncio

from pprint import pprint
from typing import Union
from random import random, randint, choice

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from exchange.exchange import Exchange, Kucoin
from exchange.util.repositories import Repository
from exchange.util.event_bus import (EventBus, LoanRepaidEvent, OrderCreatedEvent, OrderFilledEvent,
                                   OrderCancelledEvent)
from broker.models.exchange_order import build_exchange_order
from helpers.timeops import execution_time
from broker.config import CREDENTIALS


CLIENT = Kucoin(market='CROSS MARGIN', credentials=CREDENTIALS)


# ==============================================================================
"""example for how to run blocking calls asyncronously:

# we create our blocking target function for multi-threading
def blocking_func(n):
	time.sleep(0.5)
	return n ** 2


# we define our main coroutine
async def main(loop, executor):
	print('creating executor tasks')

  # create a list of coroutines and execute in the event loop
	blocking_tasks = [loop.run_in_executor(executor, blocking_func, i) for i in range(6)]
	print('waiting for tasks to complete')

  # group the results of all completed coroutines
	results = await asyncio.gather(*blocking_tasks)
	print(f'results: {results}')


if __name__ == '__main__':
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
	loop = asyncio.get_event_loop()
	try:
		loop.run_until_complete(main(loop, executor))
	finally:
		loop.close()
"""
@execution_time
def check_latency():
    res = CLIENT._get_average_latency()
    print('average roundtrip time (ms): ', res)

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

    print(f'got account with {len(acc)} balances')
    print(f'got symbols list with {len(sym)} symbols')
    print(f'got tickers list with {len(tic)} tickers')
    print(f'got orders list with {len(ord_)} orders')


def test_async_repository():
    check_latency()
    repo = get_repository()
    check_repository(repo)




# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    test_async_repository()
    # asyncio.run(amain())