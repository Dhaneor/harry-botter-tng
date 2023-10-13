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

# create a Connection object
endpoint = "/mock/websockets"
c = ws.Connection(None, endpoint, debug=True)
# create a Topics object
t = ws.Subscribers()

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


# random topic generator
def random_topic():
    return f"topic_{int(time.time() / 1000 * random() )}"


# mock publish coroutine
async def callback(msg):
    logger.info(f"received message: {msg}")


# create a WebsocketBase object
wsb = ws.WebsocketBase(callback=callback, debug=True)


# --------------------------------------------------------------------------------------
#                      test methods of TOPICS class
async def test_add_remove_topics(runs=20):
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


async def test_filter_topics():
    topics = ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7"]

    tt1, tt2, tt3 = "topic1", "topic8", "topic9"

    for topic in topics:
        await t.add_topic(topic)

    logger.debug(t._topics)

    assert tt1 in t._topics
    assert tt2 not in t._topics

    should_be_empty_list = await t.filter_topics([tt1], "add")
    assert should_be_empty_list == [], should_be_empty_list

    should_contain_tt2 = await t.filter_topics([tt1, tt2], "add")
    assert should_contain_tt2 == [tt2], should_contain_tt2

    should_contain_tt3 = await t.filter_topics([tt1, tt3], "add")
    assert should_contain_tt3 == [tt3], f"{should_contain_tt3} != {[tt3]}"


# ......................................................................................
#                    test methods of CONNECTION class
async def test_connection_prep_topic():
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


async def test_conn_watch_unwatch():
    topics = [random_topic() for _ in range(50)]

    await c.watch(topics)
    await asyncio.sleep(10)
    assert len(c._topics) == 50, f"{len(c._topics)} != 50"

    await c.unwatch(topics)
    await asyncio.sleep(10)
    assert len(c._topics) == 0, f"{len(c._topics)}!= 0"


async def test_prep_unsub_str():
    topics = [random_topic() for _ in range(50)]
    unsub_str = await c._prep_unsub_str(topics)

    logger.debug(unsub_str)


# ......................................................................................
#                  test methods of WEBSOCKETBASE class
async def test_wsb_watch_unwatch_batch():
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
    symbols = [elem["symbol"] for elem in (await exchange.fetch_markets())]
    sleep_time = 0.1
    run = 0

    while True:
        try:
            topic = "BTC/USDT"  # choice(symbols)
            runs = int(0.5 + random() * 2)
            threshhold = random() + (run / 100) - 0.5
            logger.debug("----------> %s ~ %s <----------", run, threshhold)

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
                    await asyncio.sleep(sleep_time / runs)

            run += 0.001
            await asyncio.sleep(random() * 10)

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
    await exchange.close()
    logger.info("exchange closed: OK")


async def main():
    # await test_add_remove_topics()
    # await test_filter_topics()
    # await test_connection_prep_topic()
    # await test_prep_unsub_str()
    # await test_wsb_watch_unwatch_batch()
    await test_wsb_watch_unwatch_random()
    # await test_connection_management()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutdown complete: OK")
