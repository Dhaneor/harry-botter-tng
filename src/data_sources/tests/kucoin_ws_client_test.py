#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:26:23 2023

@author_ dhaneor
"""
import asyncio
import logging
import sys

from kucoin.client import WsToken
from os.path import dirname as dir

sys.path.append(dir(dir(dir(__file__))))

from data_sources.kucoin.kucoin.ws_client import KucoinWsClient  # noqa: E402

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)


async def callback(data):
    logger.info(data)


async def main():
    wsc = await KucoinWsClient.create(
        loop=asyncio.get_event_loop(),
        client=WsToken(),
        callback=callback
    )

    logger.info("using callback: %s", wsc._callback)

    topic = "/market/ticker:BTC-USDT"

    await wsc.subscribe(topic)
    await asyncio.sleep(1)
    await wsc.subscribe(topic)
    await asyncio.sleep(5)

    logger.info("topics after 2x subscribe: %s", wsc._conn.topics)

    await wsc.unsubscribe(topic)

    logger.info("topics after 1x unsubscribe: %s", wsc._conn.topics)

    await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
