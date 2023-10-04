#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:44:23 2022

@author_ dhaneor
"""
import asyncio
import logging
import os
import sys
import zmq
import zmq.asyncio

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)


# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

from src.zmqbricks import registration as reg  # noqa: E402

SEND_ADDR = "inproc://reg_test"
RECV_ADDR = "inproc://reg_test"

test_msg = reg.Scroll(
    uid="jhfs-950746",
    name='streamer',
    service_name='test service name',
    service_type='streamer',
    endpoints={"publisher": SEND_ADDR, "management": "tcp://127.0.0.1:5600"},
    version='0.0.1',
    exchange='kucoin',
    markets=['spot'],
    description='Kucoin OHLCV streamer',
)


# ======================================================================================
async def callback(req: reg.Scroll) -> None:
    logger.info("received request: %s", req)


async def main() -> None:
    ctx = zmq.asyncio.Context()

    reg_sock = ctx.socket(zmq.ROUTER)
    reg_sock.bind(SEND_ADDR)
    client_sock = ctx.socket(zmq.DEALER)
    client_sock.connect(RECV_ADDR)

    monitor = asyncio.create_task(
        reg.monitor_registration(reg_sock, callbacks=[callback])
    )

    for _ in range(2):
        await test_msg.send(client_sock)
        await asyncio.sleep(1)

    monitor.cancel()

    await asyncio.gather(monitor)

if __name__ == '__main__':
    asyncio.run(main())
