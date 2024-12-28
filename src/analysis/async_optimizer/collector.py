#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 28 04:08:20 2024

@author dhaneor
"""

import asyncio
import zmq
import zmq.asyncio
import logging
import json

# Configure logging
logger = logging.getLogger("main.oracle")


async def oracle(context, oracle_address="tcp://*:5556", result_file="results.json"):
    """
    ZeroMQ Oracle (Sink) for collecting backtesting results from workers.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        oracle_address (str): Address to bind the Oracle's PULL socket.
        result_file (str): File to store aggregated results.
    """
    # Socket for receiving results from workers
    oracle_socket = context.socket(zmq.PULL)
    oracle_socket.bind(oracle_address)

    results = []
    errors = []

    logging.info("[ORACLE] Started and waiting for results from workers...")

    try:
        while True:
            # Receive messages from workers
            message = await asyncio.wait_for(
                oracle_socket.recv_json(),
                timeout=10
            )
            worker_id = message.get("worker", "unknown_worker")
            task = message.get("task", "unknown_task")

            if "error" in message:
                error = message["error"]
                logging.error(
                    "[ORACLE] Worker %s reported an error on task %s: %s",
                    worker_id, task, error
                )
                errors.append({"worker": worker_id, "task": task, "error": error})
            else:
                result = message["result"]
                logging.info(
                    "[ORACLE] Worker %s completed task %s with result: %s",
                    worker_id, task, result
                )
                results.append({"worker": worker_id, "task": task, "result": result})
    except asyncio.TimeoutError:
        logging.info("[ORACLE] Timeout while waiting for worker messages.")
    except asyncio.CancelledError:
        logging.info("[ORACLE] Task cancelled. Shutting down gracefully...")
    except KeyboardInterrupt:
        logging.info("[ORACLE] Interrupted by user. Shutting down...")
    finally:
        # Save results to a file
        # with open(result_file, "w") as f:
        #     json.dump({"results": results, "errors": errors}, f, indent=4)
        # logging.info(f"[ORACLE] Results saved to {result_file}")

        oracle_socket.close()
        logging.info("[ORACLE] Socket closed. Shutdown complete.")
