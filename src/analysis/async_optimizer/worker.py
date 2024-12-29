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
from .protocol import MT, ROLES, Message, Ready

# Configure logging
logger = logging.getLogger("main.worker")
logger.setLevel(logging.INFO)

TIMEOUT = 10  # Time in seconds to wait for broker messages
CHUNK_SIZE = 1_000  # Number of tasks to process in one chunk

ROLE = ROLES.WORKER

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
        # Simulate variable task duration
        process_time = random.uniform(450, 550) / 1_000_000
        time.sleep(process_time)
        results.append(random.uniform(0, 100))

        if random.random() < 0.01:  # Simulate a failure with 1% probability
            errors.append(f"Simulated failure on task {task}")
            # raise ValueError("Simulated backtest failure")

    return {"task": task, "results": results, "errors": errors}


# class Messenger:

#     def __init__(self, queue: asyncio.Queue):
#         self.queue = queue

#     async def run(self):
#         """
#         Main worker loop.

#         Pulls tasks from the broker and executes them.
#         """
#         while True:
#             task = await self.queue.get()
#             socket, message = task

#             if message is None:
#                 logger.info("Received shutdown request...")
#                 break

#             if isinstance(message, list):
#                 try:
#                     logger.debug("sending results to oracle...")
#                     await socket.send_multipart(message)
#                     self.queue.task_done()
#                 except Exception as e:
#                     logger.error(f"Error sending message: {e}")
#                     self.queue.task_done()
#             elif isinstance(message, dict):
#                 try:
#                     await socket.send_json(message)
#                     self.queue.task_done()
#                 except Exception as e:
#                     logger.error(f"Error sending message: {e}")
#                     self.queue.task_done()


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
    await asyncio.sleep(random.uniform(0.1, 2))  # Simulate worker startup delay
    # await asyncio.sleep(2)

    # Initialize sockets
    broker_socket = context.socket(zmq.DEALER)
    broker_socket.connect(broker_address)

    oracle_socket = context.socket(zmq.PUSH)
    oracle_socket.connect(oracle_address)

    q = asyncio.Queue()

    if worker_id is None:
        worker_id = f"worker-{random.randint(1000, 9999)}"

    logger.info(
        "[%s] Started and connecting to Broker (%s) and Oracle (%s)...",
        worker_id, broker_address, oracle_address
        )

    # Notify broker that the worker is ready
    register_msg = [worker_id.encode(), b"", b"HOY"]
    ready_msg = [worker_id.encode(), b"", b"READY"]
    bye_msg = [worker_id.encode(), b"", b"BYE"]

    messenger = Messenger(q)
    asyncio.create_task(messenger.run())

    execution_times = []

    await oracle_socket.send_json({"status": "READY"})
    await broker_socket.send_multipart(register_msg)

    try:
        while True:
            # Receive task from broker
            _, task = await asyncio.wait_for(
                broker_socket.recv_multipart(),
                timeout=TIMEOUT
            )

            start_time = time.time()
            task = task.decode()
            logger.debug("[%s] Received task: %s" % (worker_id, task))

            if task == "DONE":
                logger.info(f"[{worker_id}] Received DONE. Shutting down...")
                await q.put((broker_socket, bye_msg))
                await oracle_socket.send_json({"status": "DONE"})
                await q.put((None, None))  # Signal the Messenger to stop
                break

            await q.put((broker_socket, ready_msg))

            try:
                # Run the backtest
                # result = await run_backtest(task)
                result = {
                    "worker": worker_id,
                    "task": task,
                    "results": [random.uniform(0, 100) for _ in range(CHUNK_SIZE)],
                    }

            except Exception as e:
                result = {"worker": worker_id, "task": task, "error": str(e)}
            finally:
                # Notify broker that the worker is ready again
                logger.debug("[%s] Sending READY to Broker..." % worker_id)
                await q.put((oracle_socket, result))

            execution_time = (time.time() - start_time)
            execution_times.append(execution_time)
            logger.debug("[%s] time taken: %s µs" % (worker_id, execution_time))

    except asyncio.TimeoutError:
        logger.error(
            "[%s] Timeout waiting for Broker message." % worker_id
        )
    except asyncio.CancelledError:
        logger.info("[%s] Task cancelled. Shutting down..." % worker_id)
    except KeyboardInterrupt:
        logger.info("[%s] Interrupted by user. Shutting down..." % worker_id)
    finally:
        # Graceful shutdown
        broker_socket.close(1)
        oracle_socket.close()
        context.term()
        avg_exc_time = seconds_to(sum(execution_times) / (len(execution_times) + 1))
        logger.info("[%s] average execution time: %s", worker_id, avg_exc_time)
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
