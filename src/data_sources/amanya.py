#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a central confiuration service/registry for ZMQ components.

Created on Tue Sep 12 19:41:23 2023

@author_ dhaneor
"""
import asyncio
import json
import logging
import os
import sys
import zmq
import zmq.asyncio

from typing import Optional, TypeVar  # noqa: F401

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

from zmqbricks import heartbeat as hb  # noqa: F401, E402
from zmq_config import BaseConfig, Amanya  # noqa: F401, E402

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)


ConfigT = TypeVar("ConfigT", bound=BaseConfig)


# ======================================================================================
async def amanya(config: ConfigT, context: Optional[zmq.asyncio.Context] = None):
    """The Central Configuration Service (service registry)

    Parameters
    ----------
    config : ConfigT
        the configuration for the amanaya component
    ctx : zmq.asyncio.Context, optional
        A ZeroMQ Context object, default None
    """
    ctx = context or zmq.asyncio.Context()
    poller = zmq.asyncio.Poller()

    registry: dict = {}

    # configure sockets
    logger.debug("configuring heartbeat socket at %s", config.hb_addr)
    heartbeat = ctx.socket(zmq.PUB)
    heartbeat.bind(config.endpoints.get("heartbeat"))

    logger.info("configuring registration socket at %s", config.rgstr_addr)
    registration = ctx.socket(zmq.ROUTER)
    registration.curve_secretkey = config.private_key.encode("ascii")
    registration.curve_publickey = config.public_key.encode("ascii")
    registration.curve_server = True
    registration.bind(config.endpoints.get("registration"))

    logger.info("configuring requests socket at %s", config.req_addr)
    requests = ctx.socket(zmq.ROUTER)
    requests.curve_secretkey = config.private_key.encode("ascii")
    requests.curve_publickey = config.public_key.encode("ascii")
    requests.curve_server = True
    requests.bind(config.endpoints.get("requests"))

    for sock in (registration, requests):
        poller.register(sock, zmq.POLLIN)

    while True:
        try:
            logger.info("running ...")
            events = dict(await poller.poll(1000))

            if requests in events:
                msg = await requests.recv_multipart()
                key, req = msg[0], msg[1]

                reply = registry.get(req, None)
                reply = json.dumps(reply) if reply is not None else b""

                requests.send_multipart([key, reply])

        except zmq.ZMQError as e:
            logger.error(e)
        except asyncio.CancelledError:
            break

    for sock in (registration, requests, heartbeat):
        sock.close()

    if context is not None:
        ctx.term()


# ======================================================================================
async def main():
    ctx = zmq.asyncio.Context()
    config = Amanya()

    try:
        await amanya(config, ctx)
    except asyncio.CancelledError:
        logger.info("cancelled ...")

    logger.info("shutdown complete: OK")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown complete: KeyboardInterrupt")
