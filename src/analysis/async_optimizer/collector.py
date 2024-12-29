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
import time

# Configure logging
logger = logging.getLogger("main.oracle")
logger.setLevel(logging.INFO)

TIMEOUT = 5  # Time in seconds to wait for worker messages


async def oracle(context, oracle_address):
    """
    ZeroMQ Oracle (Sink) for collecting backtesting results from workers.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        oracle_address (str): Address to bind the Oracle's PULL socket.
        result_file (str): File to store aggregated results.
    """
    # Socket for receiving results from workers
    workers_socket = context.socket(zmq.PULL)
    workers_socket.bind(oracle_address)

    poller = zmq.asyncio.Poller()
    poller.register(workers_socket, zmq.POLLIN)

    results = []
    errors = []
    timestamps = []
    shutdown_requested = False

    logger.info(
        "[ORACLE] Started and waiting for results from workers at %s ..."
        % oracle_address
        )

    try:
        while not shutdown_requested:
            # Receive messages from workers
            events = await poller.poll(TIMEOUT)
            if events:
                for socket, _ in events:
                    if socket == workers_socket:
                        message = await workers_socket.recv_json()

                        if message.get("status") == "READY":
                            logger.info("received READY message")
                            continue

                        if message.get("status") == "DONE":
                            logger.info("received DONE message")
                            shutdown_requested = True
                            break

                        timestamps.append(time.time())
                        worker_id = message.get("worker", "unknown_worker")
                        task = message.get("task", "unknown_task")

                        result = message.get("results", [])
                        error = message.get("errors", [])
                        logger.debug(
                            "[ORACLE] Worker %s completed task %s "
                            "with %s results and %s errors",
                            worker_id, task, len(result), len(error)
                        )
                        results.extend(result)
                        errors.extend(error)
            if len(results) >= 1_000_000_000:
                break
    except asyncio.TimeoutError:
        logger.warning("[ORACLE] Timeout while waiting for messages - shutting down...")
    except asyncio.CancelledError:
        logger.info("[ORACLE] Task cancelled. Shutting down gracefully...")
    except KeyboardInterrupt:
        logger.info("[ORACLE] Interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"[ORACLE] An error occurred: {e}")
    else:
        if timestamps:
            now = time.time() * 1000  # milliseconds since epoch
            first_ts = timestamps[0] * 1000
            last_ts = timestamps[-1] * 1000 if len(timestamps) > 1 else now
            total_time = last_ts - first_ts
            num_elems = len(results) + len(errors) + 1
            avg_time = (last_ts - first_ts) / num_elems
            per_second = num_elems / total_time * 1000
            per_minute = per_second * 60
        else:
            total_time, avg_time, per_second, per_minute = 0, 0, 0, 0

        logger.info(f"[ORACLE] Total results: {len(results)} (errors: {len(errors)})")
        logger.info(f"[ORACLE] Processing time: {total_time:.4f} milliseconds")
        logger.info(
            "[ORACLE] Average processing time: %s milliseconds (%s/s , %s/min)",
            f"{avg_time:.2f}", f"{int(per_second):,}", f"{int(per_minute):,}"
            )
    finally:
        workers_socket.close()
        logger.info("[ORACLE] Socket closed. Shutdown complete.")
