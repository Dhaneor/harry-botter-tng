#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 16 13:09:20 2023

@author dhaneor
"""
import asyncio
import logging
import os
import sys
import zmq
import zmq.asyncio

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

from src.data_sources.collector import collector  # noqa: E402
from src.data_sources.stream_manager import stream_manager  # noqa: E402

context = zmq.asyncio.Context()


# =====================================================================================
async def stream_manager_client():
    socket = context.socket(zmq.PUSH)
    requests_address = "tcp://localhost:5597"
    socket.connect(requests_address)

    while True:
        await asyncio.sleep(2)
        logger.info("sending request to %s...", requests_address)

        await socket.send_json(
            {
                "type": "request",
                "data": {"name": "collector", "action": "get_ohlcv"},
            }
        )


async def main():
    collector_args = dict(
        ctx=context,
        subscriber_address="tcp://*:5501",
        manager_address="tcp://*:5502",
        publisher_address="tcp://*:5503",
    )

    tasks = [
        asyncio.create_task(stream_manager_client()),
        asyncio.create_task(stream_manager(ctx=context)),
        asyncio.create_task(collector(**collector_args)),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("CancelledError .. shutting down gracefully..")

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("done")


if __name__ == "__main__":
    logger.debug("starting...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug("shutdown complete: OK")

    context.term()
