#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from collections import namedtuple
import logging
import sys
import time
import zmq
import zmq.asyncio

from typing import Optional, Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Creating console handler and setting its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Creating a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Adding formatter to the console handler
ch.setFormatter(formatter)

# Adding console handler to the logger
logger.addHandler(ch)


async def sub_listener(
    ctx: zmq.asyncio.Context,
    address: str,
    topic: str,
    callback: Optional[Callable[[dict], None]] = None,
):
    """A simple subscriber that listens to a given topic.

    The messages that we receive are passed to the callback function.
    Their inner structure depends on the publisher, but they are always
    dictionaries.

    Parameters
    ----------
    ctx : zmq.asyncio.Context
        a working zmq context
    address : str
        the IP address:port of the publisher
    topic : str
        a valid topic to subscribe to
    callback : Optional[Callable[[dict], None]], optional
        a callback function/method that can handle the messages that
        we receive here, by default None
    """

    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(address)
    subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode())

    logger.info(f'subscriber started for topic {topic}')

    count = 0
    start = time.time()
    try:
        while True:
            msg = None

            try:
                msg = await subscriber.recv_multipart()
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:  # Interrupted
                    break
                else:
                    raise

            logger.info(msg[1].decode())
            count += 1

    finally:
        end = time.time()
        duration = round(end - start)
        logger.info(
            f"Subscriber received {count} messages in {duration} seconds "
            f"(={round(count / duration)} msg/s)"
        )


async def main(address, topic):
    ctx = zmq.asyncio.Context()
    logger.debug("starting zmq subscriber %s %s", address, topic)

    try:
        await sub_listener(ctx=ctx, address=address, topic=topic)
    except KeyboardInterrupt:
        ctx.term()


if __name__ == '__main__':
    try:
        address = sys.argv[1]
        if 'tcp://' not in address or len(address.split(':')) != 3:
            raise ValueError(f"Invalid address format: {address}")

        topic = sys.argv[2] if len(sys.argv) == 3 else ''
        if not isinstance(topic, str):
            raise ValueError(f"Invalid topic type: {type(topic)}")

        logger.info(f'connecting to {address} and subscribing for topic: {topic}')
        asyncio.run(main(address, topic))

    except IndexError:
        logger.error('usage: python zmq_subscriber.py "tcp://<host ip>:<port>" <topic>')
        logger.error('\\nWhen no topic is given, we subscribe to all topics.')
    except ValueError as ve:
        logger.error(ve)
    except KeyboardInterrupt:
        pass
