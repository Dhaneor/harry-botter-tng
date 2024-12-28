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

# Configure logging
logger = logging.getLogger("main.worker")
logger.setLevel(logging.ERROR)


async def run_backtest(task):
    """
    Simulate backtest execution.
    Replace this with your actual backtesting logic.

    Args:
        task (str): The task/parameters for backtesting.

    Returns:
        dict: Results of the backtest or raises an exception.
    """
    if random.random() < 0.1:  # Simulate a failure with 10% probability
        raise ValueError("Simulated backtest failure")

    await asyncio.sleep(random.uniform(0.1, 3))  # Simulate variable task duration
    return {"task": task, "result": random.uniform(0, 100)}


async def worker(
    context,
    worker_id=None,
    broker_address="tcp://localhost:5555",
    oracle_address="tcp://localhost:5556",
):
    """
    Async ZeroMQ Worker for executing backtesting tasks.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        worker_id (str): Unique identifier for the worker.
        broker_address (str): Address of the broker.
        oracle_address (str): Address of the Oracle (Sink).
    """
    await asyncio.sleep(random.uniform(0.1, 0.2))  # Simulate worker startup delay

    # Initialize sockets
    broker_socket = context.socket(zmq.DEALER)
    broker_socket.connect(broker_address)

    oracle_socket = context.socket(zmq.PUSH)
    oracle_socket.connect(oracle_address)

    if worker_id is None:
        worker_id = f"worker-{random.randint(1000, 9999)}"

    logging.info(f"[{worker_id}] Started and connecting to Broker and Oracle...")

    # Notify broker that the worker is ready
    ready_msg = [worker_id.encode(), b"", b"READY"]
    bye_msg = [worker_id.encode(), b"", b"BYE"]
    await broker_socket.send_multipart(ready_msg)

    try:
        while True:
            # Receive task from broker
            _, task = await broker_socket.recv_multipart()
            logger.info(f">>>>>>>>>>>>>>>>> [{worker_id}] Received task: {task}")
            task = task.decode()

            if task == "DONE":
                logger.info(f"[{worker_id}] Received DONE. Shutting down...")
                break

            logger.info(f"[{worker_id}] Received task: {task}")

            try:
                # Run the backtest
                result = await run_backtest(task)
                logger.info(f"[{worker_id}] Task completed: {result}")
                await oracle_socket.send_json(
                    {"worker": worker_id, "task": task, "result": result}
                )

            except Exception as e:
                # Send failure notice to Oracle
                await oracle_socket.send_json(
                    {"worker": worker_id, "task": task, "error": str(e)}
                )
                logging.info(f"[{worker_id}] Task failed: {e}")

            # Notify broker that the worker is ready again
            logger.info(f"[{worker_id}] Sending READY to Broker...")
            await broker_socket.send_multipart(ready_msg)

    except asyncio.CancelledError:
        logging.info(f"[{worker_id}] Task cancelled. Shutting down...")
    except KeyboardInterrupt:
        logging.info(f"[{worker_id}] Interrupted by user. Shutting down...")

    finally:
        # Graceful shutdown
        await broker_socket.send_multipart(bye_msg)
        broker_socket.close()
        oracle_socket.close()
        logging.info(f"[{worker_id}] Shutdown complete.")
