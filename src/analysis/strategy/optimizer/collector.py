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
from time import time

from .protocol import TYPE, ROLES, Message
from util import seconds_to

# Configure logging
logger = logging.getLogger("main.oracle")
logger.setLevel(logging.INFO)

NAME = "ORACLE"
ROLE = ROLES.COLLECTOR
TIMEOUT = 1  # Time in seconds to wait for worker messages


class ResultProcessor:
    def __init__(self):
        self.results = []
        self.errors = []
        self.timestamps = []

    async def process_result(self, msg: Message) -> None:
        self.timestamps.append(time())

        if not (batch := msg.payload if msg.payload else {}):
            logger.error("[%s] Empty batch from %s %s", NAME, msg.role, msg.origin)
            return

        self.results.extend(batch.get("results", []))
        self.errors.extend(batch.get("errors", []))

    async def log_results(self) -> None:
        ok, fail = len(self.results), len(self.errors)
        total_results = ok + fail
        total_time, avg_time, per_second, per_minute = 0, 0, 0, 0

        if self.timestamps:
            first_ts = self.timestamps[0]
            last_ts = self.timestamps[-1] if len(self.timestamps) > 1 else time()
            total_time = last_ts - first_ts

            avg_time = seconds_to(total_time / max(total_results, 1))
            per_second = int(total_results / total_time)
            per_minute = per_second * 60

        logger.info(f"[{NAME}] results: {ok:,} (errors: {fail:,})")
        logger.info(f"[{NAME}] Total processing time: {seconds_to(total_time)}")
        logger.info(
            "[ORACLE] Average processing time: %s (%s/s , %s/min)",
            avg_time, f"{per_second:,}", f"{per_minute:,}"
        )


async def oracle(context, oracle_address):
    """
    ZeroMQ Oracle (Sink) for collecting backtesting results from workers.

    Args:
        context (zmq.asyncio.Context): Shared ZeroMQ async context.
        oracle_address (str): Address to bind the Oracle's PULL socket.
        result_file (str): File to store aggregated results.
    """
    workers_socket = context.socket(zmq.PULL)
    workers_socket.bind(oracle_address)

    poller = zmq.asyncio.Poller()
    poller.register(workers_socket, zmq.POLLIN)

    processor = ResultProcessor()

    known_producers = set()
    go_home = False

    logger.debug("[%s] Waiting for results at %s ..." % (NAME, oracle_address))

    try:
        while True:
            # Receive messages from workers
            events = await poller.poll(TIMEOUT)
            if events:
                for socket, _ in events:
                    if socket == workers_socket:
                        msg = Message.from_multipart(
                            await workers_socket.recv_multipart()
                        )

                        match msg.type:
                            case TYPE.RESULT:
                                await processor.process_result(msg)

                            case TYPE.HOY:
                                known_producers.add(msg.origin)

                            case TYPE.BYE:
                                known_producers.remove(msg.origin)
                                logger.debug(
                                    "[%s] BYE message from %s %s (known: %s)",
                                    NAME, msg.role, msg.origin, len(known_producers)
                                )
                                go_home = True if len(known_producers) == 0 else False

                            case _:
                                logger.error(
                                    "[%s] Received invalid message type %s from %s %s"
                                    % (NAME, msg.type.name, msg.role.name, msg.origin)
                                )
            else:
                if go_home:
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
        await processor.log_results()
    finally:
        workers_socket.close(0.2)
        logger.info("[ORACLE] Socket closed. Shutdown complete.")
