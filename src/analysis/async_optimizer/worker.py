#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 28 03:46:20 2024

@author dhaneor
"""

import asyncio
import zmq
import zmq.asyncio
import logging
import random
import time
from typing import Sequence

from util import seconds_to
from .messenger import Messenger
from .protocol import TYPE, ROLES, Message

# Configure logging
logger = logging.getLogger("main.worker")
logger.setLevel(logging.INFO)

TIMEOUT = 10  # Time in seconds to wait for broker messages
CHUNK_SIZE = 1_000  # Number of tasks to process in one chunk
DURATION = 500  # time for each backtest in microseconds

ROLE = ROLES.WORKER


def backtest():
    process_time = random.uniform(DURATION - 50, DURATION + 50) / 1_000_000
    time.sleep(process_time)
    return random.uniform(0, 100)


async def run_backtest(task, chunk_length=CHUNK_SIZE):
    """
    Simulate backtest execution.
    Replace this with your actual backtesting logic.

    Args:
        task (str): The task/parameters for backtesting.

    Returns:
        dict: Results of the backtest or raises an exception.
    """
    results = []
    errors = []
    for _ in range(chunk_length):
        if random.random() < 0.01:  # Simulate a failure with 1% probability
            _ = backtest()
            errors.append(f"Simulated failure on task {task}")
        else:
            results.append(backtest())

    return {"task": task, "results": results, "errors": errors}


async def worker(
    context: zmq.asyncio.Context,
    worker_id: str | None,
    broker_address: str,
    oracle_address: str,
):
    """
    Async ZeroMQ Worker for executing backtesting tasks.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        worker_id (str): Unique identifier for the worker.
        broker_address (str): Address of the broker.
        oracle_address (str): Address of the Oracle (Sink).
    """
    logger.info("[%s] Connecting to Broker (%s)...", worker_id, broker_address)
    # await asyncio.sleep(random.uniform(0.1, 2))  # Simulate worker startup delay
    await asyncio.sleep(0.2)

    # Initialize sockets
    broker_socket = context.socket(zmq.DEALER)
    broker_socket.connect(broker_address)

    oracle_socket = context.socket(zmq.PUSH)
    oracle_socket.connect(oracle_address)

    if worker_id is None:
        worker_id = f"worker-{random.randint(1000, 9999)}"

    logger.info(
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

    logger.info(f"[{worker_id}] Received READY from Broker and Oracle...")

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
                logger.debug("[%s] Received task: %s" % (worker_id, task))

                try:
                    result = await run_backtest(task)
                    # result = {
                    #     "worker": worker_id,
                    #     "task": task,
                    #     "results": [random.uniform(0, 100) for _ in range(CHUNK_SIZE)],
                    #     }
                except Exception as e:
                    logger.error(e)
                    result = {"worker": worker_id, "task": task, "error": str(e)}
                else:
                    await oracle.send_result(result=result)
                    # await oracle_q.put((result, None))

            # stop operation if received BYE message
            elif msg.type == TYPE.BYE:
                logger.info(
                    "[%s] Received BYE from %s. Shutting down..."
                    % (worker_id, msg.origin)
                    )
                await broker.say_goodbye()
                await oracle.say_goodbye()
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
        await broker.stop()  # Signal the Messenger to stop
        await oracle.stop()  # Signal the Messenger to stop
        await asyncio.sleep(0.1)  # Wait for all tasks to finish
        broker_socket.close()
        oracle_socket.close(1)
        context.term()
        avg_exc_time = seconds_to(sum(execution_times) / (len(execution_times) + 1))
        logger.debug("[%s] average execution time: %s", worker_id, avg_exc_time)
        logger.info(f"[{worker_id}] Shutdown complete.")


async def workers(
    context: zmq.asyncio.Context,
    worker_ids: Sequence[str],
    broker_address: str,
    oracle_address: str,
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
    logger.info(f"[MAIN] Starting {num_workers} worker processes...")
    tasks = []
    for worker_id in worker_ids:
        tasks.append(worker(context, worker_id, broker_address, oracle_address))
    await asyncio.gather(*tasks)
