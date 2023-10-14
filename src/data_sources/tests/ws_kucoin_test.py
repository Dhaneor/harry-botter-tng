#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:44:23 2023

@author_ dhaneor
"""
import asyncio
import ccxt.pro as ccxt
import logging
import sys
import time

from os.path import dirname as dir
from random import choice, random

sys.path.append(dir(dir(dir(__file__))))

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)

import data_sources.websockets.ws_kucoin as ws  # noqa: E402


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

    return symbols[:count]


# random topic generator
def random_topic():
    return f"topic_{int(time.time() / 1000 * random() )}"


# mock publish coroutine
async def callback(msg):
    pass
    # logger.info(f"received message: {msg}")


# --------------------------------------------------------------------------------------
#                     test methods of SUBSCRIBERS class
async def test_add_remove_topics(runs=20):
    # create a Topics object
    t = ws.Subscribers()
    topics = ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7"]

    for _ in range(runs):
        topic = choice(topics)

        if random() < 0.5:
            subs_for_topic = t._topics.get(topic, 0)
            await t.add_topic(topic)
            assert subs_for_topic + 1 == t._topics.get(topic, 0)
        else:
            subs_for_topic = t._topics.get(topic, 0)
            await t.remove_topic(topic)
            if subs_for_topic > 1:
                assert subs_for_topic - 1 == t._topics.get(topic, 0)
            else:
                assert t._topics.get(topic, None) is None


async def test_get_item():
    # create a Topics object
    t = ws.Subscribers()

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


# ......................................................................................
#                    test methods of CONNECTION class
async def test_connection_prep_topic():
    # create a Connection object
    endpoint = "/mock/websockets"
    c = ws.Connection(None, endpoint, debug=True)

    c._topics = {i: 1 for i in range(280)}

    max_topics_per_conn = ws.MAX_TOPICS_PER_CONNECTION - 1
    no_of_topics = 50
    topics = ["topic"] * no_of_topics
    max_left = c.max_topics_left

    should_be_too_much = no_of_topics - max_left

    topics, too_much = await c._prep_topic_str(topics)
    topics = [sub_t for t in topics for sub_t in t.split(",")]
    c._topics = c._topics | {i + 500: t for i, t in enumerate(topics)}

    assert len(c._topics) == max_topics_per_conn, \
        f"{len(c._topics)} <-> {max_topics_per_conn}"
    assert len(too_much) == should_be_too_much

    # ........................................
    topics, too_much = await c._prep_topic_str("extra topic")

    assert topics == []
    assert too_much == ["extra topic"]

    logger.debug("test passed: OK")


async def test_prep_unsub_str():
    # create a Connection object
    endpoint = "/mock/websockets"
    c = ws.Connection(None, endpoint, debug=True)

    topics = [random_topic() for _ in range(50)]
    unsub_str = await c._prep_unsub_str(topics)

    logger.debug(unsub_str)


async def test_conn_watch_unwatch():
    # create a Connection object
    endpoint = "/mock/websockets"
    c = ws.Connection(None, endpoint, debug=True)
    no_of_topics = 3

    topics = [random_topic() for _ in range(no_of_topics)]

    await c.watch(topics)
    await asyncio.sleep(1)
    assert len(c._topics) == no_of_topics, f"{len(c._topics)} != 50"

    await c.unwatch(topics)
    await asyncio.sleep(1)
    assert len(c._topics) == 0, f"{len(c._topics)}!= 0 --> {c._topics}"


# ......................................................................................
#                  test methods of WEBSOCKETBASE class
async def test_filter_topics():
    wsb = ws.WebsocketBase(callback=callback, debug=True)
    topics = [random_topic() for _ in range(10)]
    logger.debug(topics)

    topics = await wsb.filter_existing_topics(topics, "subscribe")
    logger.debug(topics)

    assert len(topics) == 10, f"{len(topics)}!= 10"

    await wsb.watch(topics)
    await asyncio.sleep(2)
    await wsb.close()
    await asyncio.sleep(2)


async def test_wsb_watch_unwatch_batch():
    wsb = ws.WebsocketBase(callback=callback, debug=True)
    topics = [random_topic() for _ in range(50)]

    await wsb.watch(topics)
    await asyncio.sleep(10)

    for conn in wsb._connections.values():
        logger.debug("%s has topics: %s", conn.name, len(conn._topics))

    # assert len(wsb._topics) == 50, f"{len(c._topics)} != 50"

    logger.debug("-~•~-" * 50)

    await wsb.unwatch(topics)
    await asyncio.sleep(20)
    # assert len(wsb._topics) == 0, f"{len(c._topics)}!= 0"

    logger.debug("test passed: OK")


async def test_wsb_watch_unwatch_random():
    # create a WebsocketBase object
    wsb = ws.WebsocketBase(callback=callback, debug=True)
    symbols = await get_symbols(50)
    sleep_time = 0.1
    run = 0

    while True:
        try:
            topic = choice(symbols)
            runs = int(0.5 + random() * 2)
            threshhold = random() + (run / 100) - 0.5
            # logger.debug("----------> %s ~ %s <----------", run, threshhold)

            if random() > threshhold:
                for _ in range(runs):
                    await wsb.watch(topic)
                    await asyncio.sleep(sleep_time / runs)

            else:
                for _ in range(runs):
                    if (subbed_topics := list(wsb.topics.keys())):
                        await wsb.unwatch(choice(subbed_topics))
                    else:
                        logger.debug("no topics to unwatch")
                        if run > 20:
                            raise asyncio.CancelledError()
                    await asyncio.sleep(sleep_time / runs)

            run += 1
            # await asyncio.sleep(random() * 1)

        except TypeError as e:
            logger.error(e, exc_info=1)
            logger.debug(type(wsb.topics))
        except asyncio.CancelledError:
            logger.info("test cancelled ...")
            break
        except Exception as e:
            logger.error(e, exc_info=1)
            break
    await wsb.close()
    logger.info("exchange closed: OK")


async def test_it_for_real():
    """Test with the real websocket client."""
    wsb = ws.WsTickers(callback=callback)
    symbols = await get_symbols(1)
    sleep_time = 2
    run = 0

    while True:
        try:
            topic = choice(symbols)
            runs = int(0.5 + random() * 2)
            threshhold = random() + (run / 50) - 0.5
            # logger.debug("----------> %s ~ %s <----------", run, threshhold)

            if random() > threshhold:
                for _ in range(runs):
                    await wsb.watch(topic)
                    await asyncio.sleep(sleep_time / runs)

            else:
                for _ in range(runs):
                    if (subbed_topics := list(wsb.topics.keys())):
                        await wsb.unwatch(choice(subbed_topics))
                        await asyncio.sleep(sleep_time / runs)
                    else:
                        logger.debug("no topics to unwatch")
                        if run > 20:
                            raise asyncio.CancelledError()

            run += 1
            await asyncio.sleep(random() * 1)

        except TypeError as e:
            logger.error(e, exc_info=1)
            logger.debug(type(wsb.topics))
        except asyncio.CancelledError:
            logger.info("test cancelled ...")
            break
        except Exception as e:
            logger.error(e, exc_info=1)
            break
    await wsb.close()
    logger.info("exchange closed: OK")


async def main():
    # await test_add_remove_topics()
    # await test_get_item()

    # await test_connection_prep_topic()
    # await test_conn_watch_unwatch()
    # await test_prep_unsub_str()

    # await test_filter_topics()
    # await test_wsb_watch_unwatch_batch()
    await test_wsb_watch_unwatch_random()
    # await test_it_for_real()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutdown complete: OK")
