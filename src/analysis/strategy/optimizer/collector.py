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

from .protocol import TYPE, ROLES, Message
from util import seconds_to

# Configure logging
logger = logging.getLogger("main.oracle")
logger.setLevel(logging.INFO)

NAME = "ORACLE"
ROLE = ROLES.COLLECTOR
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

    known_producers = set()
    results = []
    errors = []
    timestamps = []
    shutdown_requested = False

    logger.debug(
        "[%s] Started and waiting for results from workers at %s ..."
        % (NAME, oracle_address)
    )

    try:
        while not shutdown_requested:
            # Receive messages from workers
            events = await poller.poll(TIMEOUT)
            if events:
                for socket, _ in events:
                    if socket == workers_socket:
                        msg = Message.from_multipart(
                            await workers_socket.recv_multipart()
                        )

                        logger.debug(
                            "received message from worker %s: %s"
                            % (msg.origin, msg.type.name)
                        )

                        match msg.type:
                            case TYPE.HOY:
                                known_producers.add(msg.origin)

                            case TYPE.READY:
                                logger.info("received READY message")
                                continue

                            case TYPE.BYE:
                                known_producers.remove(msg.origin)
                                logger.debug(
                                    "[ORACLE] received BYE message from "
                                    "worker %s (known: %s)"
                                    % (msg.origin, len(known_producers))
                                )

                                if len(known_producers) == 0:
                                    shutdown_requested = True
                                    break

                            case TYPE.RESULT:
                                timestamps.append(time.time())

                                logger.debug(
                                    "[ORACLE] Received %s results from worker %s"
                                    % (len(msg.payload.get("results", [])), msg.origin)
                                )

                                worker_id = msg.origin
                                batch = msg.payload if msg.payload else {}

                                if not batch:
                                    logger.error(
                                        "[ORACLE] Received invalid batch from worker %s"
                                        % worker_id
                                    )
                                    continue

                                task = batch.get("task", "unknown_task")
                                batch_results = batch.get("results", None)
                                batch_errors = batch.get("errors", [])
                                logger.debug(
                                    "[ORACLE] Worker %s completed task %s "
                                    "with %s results and %s errors",
                                    worker_id,
                                    task,
                                    len(batch_results),
                                    len(batch_errors),
                                )
                                results.extend(batch_results)
                                errors.extend(batch_errors)
                            case _:
                                logger.error(
                                    "[%s] Received invalid message type %s from %s %s"
                                    % (NAME, msg.type.name, msg.role.name, msg.origin)
                                )
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
            now = time.time()
            first_ts = timestamps[0]
            last_ts = timestamps[-1] if len(timestamps) > 1 else now
            total_time = last_ts - first_ts
            num_elems = len(results) + len(errors)
            avg_time = total_time / (num_elems if num_elems > 0 else 1)
            per_second = num_elems / total_time
            per_minute = per_second * 60
        else:
            total_time, avg_time, per_second, per_minute = 0, 0, 0, 0

        logger.info(
            f"[ORACLE] Total results: {len(results):,} (errors: {len(errors):,})"
        )
        logger.info(f"[ORACLE] Processing time: {seconds_to(total_time)}")
        logger.info(
            "[ORACLE] Average processing time: %s (%s/s , %s/min)",
            seconds_to(avg_time),
            f"{int(per_second):,}",
            f"{int(per_minute):,}",
        )
    finally:
        workers_socket.close()
        logger.info("[ORACLE] Socket closed. Shutdown complete.")
