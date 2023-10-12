#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:44:23 2023

@author_ dhaneor
"""
import asyncio
import logging
import sys
import time

from os.path import dirname as dir
from pprint import pprint
from random import choice, random

sys.path.append(dir(dir(dir(__file__))))
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')

import data_sources.websockets.ws_kucoin as ws  # noqa: E402

# create a Connection object
c = ws.Connection(None, True)
# create a Topics object
t = ws.Topics()


# --------------------------------------------------------------------------------------
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


async def test_wait_for():
    for _ in range(ws.MSG_LIMIT):
        c._ts_msgs.append(time.time())
        await asyncio.sleep(0.1)

    for _ in range(100):
        oldest = round(time.time() - min(c._ts_msgs), 2)
        wf = await c._wait_for()
        no_of_msgs = len(c._ts_msgs)
        await asyncio.sleep(random() / 10)
        logger.debug("wait for %s (%s - %s)", wf, oldest, no_of_msgs)
        c._ts_msgs.append(time.time())


async def test_connection_management():
    ...


async def main():
    # await test_add_remove_topics()
    # await test_filter_topics()
    # await test_connection_prep_topic()
    # await test_connection_management()
    await test_wait_for()

if __name__ == "__main__":
    asyncio.run(main())
