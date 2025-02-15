#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 28 03:46:20 2024

@author dhaneor
"""

import asyncio
import numpy as np
import zmq
import zmq.asyncio
import logging
import random
import time
from typing import Sequence, Coroutine

from util import seconds_to
from .messenger import Messenger
from .protocol import TYPE, ROLES, Message

from data.data_models import Ohlcv
from analysis.strategy_builder import build_strategy
from analysis.strategy.definitions import s_aroon_osc, s_linreg  # noqa: F401
from analysis import strategy_backtest as bt

"""
NOTE:For development purposes, we are using a specific
strategy. This logic needs to be extended and replaced
by the possibility to receive strategy names from the
broker and build and execute the corresponding strategy.
"""
DEV_STRATEGY = build_strategy(s_linreg)

# Configure logging
logger = logging.getLogger("main.worker")
logger.setLevel(logging.INFO)

TIMEOUT = 100  # Time in seconds to wait for broker messages
CHUNK_SIZE = 1000  # Number of tasks to process in one chunk
DURATION = 500  # time for each backtest in microseconds

ROLE = ROLES.WORKER


async def fetch_ohlcv(
    repo_socket: zmq.asyncio.Socket,
    request: dict
) -> dict[str, np.ndarray]:
    await repo_socket.send_json(request)
    response = Ohlcv.from_json(await repo_socket.recv_string())
    logger.debug(f"Received OHLCV data for {request['symbol']} from repository")
    return response.to_dict()


def backtest():
    process_time = random.uniform(DURATION - 50, DURATION + 50) / 1_000_000
    time.sleep(process_time)
    return random.uniform(0, 100)


def backtest_closure(worker_id: str, repo_socket: zmq.asyncio.Socket) -> Coroutine:
    worker_id = str(worker_id)
    repo_socket: zmq.asyncio.Socket

    last_task: dict = {}
    current_strategy: str = None

    ohlcv_data = {}

    async def run_backtest(task, chunk_length=CHUNK_SIZE):
        """
        Simulate backtest execution.
        Replace this with your actual backtesting logic.

        Args:
            task (str): The task/parameters for backtesting.

        Returns:
            dict: Results of the backtest or raises an exception.
        """
        nonlocal last_task
        nonlocal current_strategy
        nonlocal ohlcv_data

        logger.debug("[%s] Running backtest for task: %s" % (worker_id, task))

        if task.get('ohlcv_request') == last_task.get('ohlcv_request', None):
            logger.debug(
                "[%s] Using cached OHLCV data for %s (%s)"
                % (worker_id, task['ohlcv_request'], len(ohlcv_data))
                )
        else:
            logger.debug(
                "[%s] Fetching new OHLCV data for %s"
                % (worker_id, task['ohlcv_request']['symbol'])
                )
            ohlcv_data = await fetch_ohlcv(repo_socket, task.get('ohlcv_request'))

        data = ohlcv_data

        if task != last_task:
            last_task = task
            strategy_name = task.get("strategy")
            parameters = task.get("parameters")  # noqa: F841
            current_strategy = strategy_name
            strategy = DEV_STRATEGY
            # ... add logic to change the strategy here for real use

        else:
            strategy = DEV_STRATEGY

        results, errors, error_logged = [], [], False

        if not data:
            raise ValueError("No OHLCV data received")

        for _ in range(chunk_length):
            try:
                bt_result = bt.run(
                    strategy=strategy,
                    data=data,
                    initial_capital=10_000,
                    risk_level=7,
                    max_leverage=1
                    )
                results.append(bt_result.get('b.value')[-1])
            except Exception as e:
                if not error_logged:
                    logger.error("[%s] Error running backtest: %s", worker_id, str(e))
                    error_logged = True
                errors.append(str(e))

        return {"task": task, "results": results, "errors": errors}
    return run_backtest


async def worker(
    ctx: zmq.asyncio.Context,
    worker_id: str | None,
    broker_address: str,
    oracle_address: str,
    ohlcv_repository_address: str,
):
    """
    Async ZeroMQ Worker for executing backtesting tasks.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        worker_id (str): Unique identifier for the worker.
        broker_address (str): Address of the broker.
        oracle_address (str): Address of the Oracle (Sink).
    """
    logger.debug("[%s] Connecting to Broker (%s)...", worker_id, broker_address)

    worker_id = worker_id or f"{random.randint(1000, 9999)}"

    # Initialize sockets
    broker_socket = ctx.socket(zmq.DEALER)
    broker_socket.connect(broker_address)

    oracle_socket = ctx.socket(zmq.PUSH)
    oracle_socket.connect(oracle_address)

    repo_socket = ctx.socket(zmq.REQ)
    repo_socket.connect(ohlcv_repository_address)

    backtest_fn = backtest_closure(worker_id, repo_socket)

    logger.debug(
        "[%s] Started and connecting to Broker (%s) and Oracle (%s)...",
        worker_id, broker_address, oracle_address
        )

    # set up the messengers to broker and oracle
    broker_q = asyncio.Queue()
    oracle_q = asyncio.Queue()
    broker = Messenger(worker_id, ROLE, broker_socket, broker_q)
    asyncio.create_task(broker.run())
    oracle = Messenger(worker_id, ROLE, oracle_socket, oracle_q)
    asyncio.create_task(oracle.run())

    await broker.say_hoy()
    await oracle.say_hoy()

    execution_times = []

    try:
        while True:
            # Receive task from broker
            msg = await asyncio.wait_for(
                broker_socket.recv_multipart(),
                timeout=TIMEOUT
            )
            msg = Message.from_multipart(msg)

            start_time = time.time()

            # perform task if we got one
            if msg.type == TYPE.TASK:
                # signal READY to already request the next task
                # from broker before processing the current one
                logger.debug("[%s] Sending READY to Broker..." % worker_id)
                await broker.say_ready()
                # await broker_q.put(("READY", None))
                # await asyncio.sleep(0)  # wait for a little bit

                task = msg.payload

                task = {
                    'strategy': 'dummy',
                    'parameters': tuple(),
                    'ohlcv_request': {
                        'exchange': 'binance',
                        'symbol': 'BTC/USDT',
                        'interval': '1d',
                        'start': '2021-11-30 00:00:00 UTC',
                        'end': '2024-12-01 00:00:00 UTC'
                    }
                }

                logger.debug("[%s] Received task: %s" % (worker_id, task))

                try:
                    result = await backtest_fn(task)
                    logger.debug(
                        "[%s] Received result for task %s from backtest"
                        % (worker_id, task)
                        )
                except Exception as e:
                    logger.error(e)
                    result = {"worker": worker_id, "task": task, "error": str(e)}
                else:
                    await oracle.send_result(result=result)
                    # await oracle_q.put((result, None))

            # stop operation if received BYE message
            elif msg.type == TYPE.BYE:
                logger.debug(
                    "[%s] Received BYE from %s. Shutting down..."
                    % (worker_id, msg.origin)
                    )
                await repo_socket.send_json({"action": "close"})
                await broker.say_goodbye()
                await oracle.say_goodbye()
                logger.debug("[%s] Sent BYE to everyone. Exiting..." % worker_id)
                break

            # log an error and do nothing if we get other message types
            else:
                logger.error(
                    "[%s] Received unknown message type: %s" % (worker_id, msg.type)
                )

            execution_time = (time.time() - start_time)
            execution_times.append(execution_time)
            logger.debug(
                "[%s] message processed in : %s"
                % (worker_id, seconds_to(execution_time))
                )

    except asyncio.TimeoutError:
        logger.error(
            "[%s] Timeout waiting for Broker message." % worker_id
        )
        await oracle.say_goodbye()
    except asyncio.CancelledError:
        logger.info("[%s] Task cancelled. Shutting down..." % worker_id)
    except KeyboardInterrupt:
        logger.info("[%s] Interrupted by user. Shutting down..." % worker_id)
    finally:
        await broker.stop()  # Signal the Broker Messenger to stop
        await oracle.stop()  # Signal the Oracle Messenger to stop
        await asyncio.sleep(0.2)  # Wait for all tasks to finish
        logger.debug("[%s] Messengers stopped. Closing sockets...", worker_id)
        broker_socket.close(0.5)
        oracle_socket.close(0.5)
        repo_socket.close(0.5)
        logger.debug("[%s] broker socket closed: %s.", worker_id, broker_socket.closed)
        logger.debug("[%s] oracle socket closed: %s.", worker_id, oracle_socket.closed)
        # ctx.term()
        # logger.debug("[%s] ZMQ Context termintated.")
        avg_exc_time = seconds_to(sum(execution_times) / (len(execution_times) + 1))
        logger.debug("[%s] average execution time: %s", worker_id, avg_exc_time)
        logger.debug(f"[{worker_id}] Shutdown complete.")
        return


async def workers(
    context: zmq.asyncio.Context,
    worker_ids: Sequence[str],
    broker_address: str,
    oracle_address: str,
    ohlcv_repository_address: str,
    num_workers: int,
):
    """
    Run multiple workers in a single process.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        worker_id (str): Unique identifier for the worker.
        broker_address (str): Address of the broker.
        oracle_address (str): Address of the Oracle (Sink).
        num_workers (int): Number of workers to start.
    """
    logger.debug("[MAIN] Starting worker process with %s parallel workers ...")
    tasks = []
    for worker_id in worker_ids:
        tasks.append(
            worker(
                context,
                worker_id,
                broker_address,
                oracle_address,
                ohlcv_repository_address
                )
            )
    await asyncio.gather(*tasks)
