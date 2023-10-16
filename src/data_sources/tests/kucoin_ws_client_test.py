#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:26:23 2023

@author_ dhaneor
"""
import asyncio
import ccxt.pro as ccxt
import logging
import sys
import time

from kucoin.client import WsToken
from os.path import dirname as dir
from random import random, choice  # noqa: E402

sys.path.append(dir(dir(dir(__file__))))

from data_sources.kucoin.kucoin import ws_client as wsc  # noqa: E402
from data_sources.kucoin.kucoin.websocket.websocket import ConnectWebsocket  # noqa: E402
from data_sources.kucoin.kucoin.ws_token.token import GetToken  # noqa: E402

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)


# symbols downloader
async def get_symbols(count):
    #  create a CCXT instance for symbol names download
    exchange = ccxt.kucoin({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "fetchMarkets": True,
            "fetchOHLCV": True,
            "fetchOrderBook": True,
            "fetchTicker": True,
            "fetchTickers": True,
            "fetchTrades": True,
        },
        "verbose": False,
    })

    symbols = [elem["symbol"] for elem in (await exchange.fetch_markets())]
    symbols = [s.replace("/", "-") for s in symbols]

    await exchange.close()

    return [choice(symbols) for _ in range(count)]


# random topic generator
def random_topic():
    return f"topic_{int(time.time() / 1000 * random() )}"


# mock publish coroutine
async def callback(msg):
    logger.info(f"received message: {msg}")


# --------------------------------------------------------------------------------------
#                     test methods of SUBSCRIBERS class
async def test_add_remove_topics(runs=20):
    # create a Topics object
    t = wsc.Topics()
    topics = ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7"]
    subjects = ["ticker", "depth", "trades", "klines"]

    for _ in range(runs):
        topic = f"{choice(subjects)}:{choice(topics)}"
        subs_for_topic = await t.subscribers(topic)

        if random() < 0.5:
            await t.add_subscriber(topic)
            assert subs_for_topic + 1 == await t.subscribers(topic)

        else:
            if subs_for_topic:
                await t.remove_subscriber(topic)
                if subs_for_topic >= 1:
                    assert subs_for_topic - 1 == await t.subscribers(topic)
                else:
                    assert await t.subscribers(topic) == 0, \
                        f"{await t.subscribers(topic)} != 0"

    logger.info("test passed: OK")


async def test_batched_topics():
    wsc.MAX_BATCH_SUBSCRIPTIONS = 10
    t = wsc.Topics()
    topics = [random_topic() for _ in range(95)]

    batched_topics = await t.batch_topics(topics)
    assert len(batched_topics) == 10
    for row in batched_topics:
        print(row)

    batched_topics_str = await t.batch_topics_str(topics)
    assert len(batched_topics_str) == 10
    for row in batched_topics_str:
        print(row)


async def test_get_item():
    # create a Topics object
    t = wsc.Topics()

    topic = random_topic()
    await t.add_subscriber(topic)
    assert topic in t._topics
    assert t._topics[topic] == 1

    await t.add_subscriber(topic)
    assert t._topics[topic] == 2

    logger.debug(t[topic])

    await t.remove_subscriber("not_there")

    await t.remove_subscriber(topic)
    assert t._topics[topic] == 1

    await t.remove_subscriber(topic)
    assert topic not in t._topics


# --------------------------------------------------------------------------------------
#                     test methods of KucoinWsClient class
async def test_multiple_unsubscribe():
    client = await wsc.KucoinWsClient.create(
        loop=asyncio.get_event_loop(),
        client=WsToken(),
        callback=callback
    )

    logger.info("using callback: %s", wsc._callback)

    topic = "/market/ticker:BTC-USDT"

    await client.subscribe(topic)
    await asyncio.sleep(1)
    await client.subscribe(topic)
    await asyncio.sleep(5)

    logger.info("topics after 2x subscribe: %s", wsc._conn.topics)

    await client.unsubscribe(topic)

    logger.info("topics after 1x unsubscribe: %s", wsc._conn.topics)

    await asyncio.sleep(10)


async def main():
    try:
        await test_batched_topics()
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    asyncio.run(main())
