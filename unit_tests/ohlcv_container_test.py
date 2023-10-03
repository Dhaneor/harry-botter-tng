#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 20:15:20 2023

@author dhaneor
"""
import asyncio
import json
import os
import sys
import time
import logging
import zmq
import zmq.asyncio

from cProfile import Profile  # noqa E402
from pstats import SortKey, Stats  # noqa E402

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

from src.data_sources.craeft_pond import (  # noqa E402
    craeft_pond, Container, get_initial_data,
)
# from src.data_sources.listener import make_candle_event  # noqa E402
from src.data_sources.ohlcv_repository import ohlcv_repository  # noqa E402
from src.data_sources.zmq_config import OhlcvRegistry, Collector  # noqa E402

repo_addr = "inproc://ohlcv_repository"


# get the initial OHLCV data
# initial_data = None
# initial_data = get_ohlcv(symbol='ETHUSDT', interval='1d')
# del initial_data['human open time']


# example update message
msg = {
    "exchange": "binance",
    "subject": "candles",
    "topic": "ETH/USDT_1min",
    "type": "update",
    "symbol": "ETH/USDT",
    "interval": "1m",
    "data": {
        "open time": "1694369880000",
        "open": "1612.87",
        "high": "1613.06",
        "low": "1612.5",
        "close": "1612.63",
        "volume": "2.4231162",
        "quote volume": "3908.049934591"
    },
    "time": 1694369937.3175936,
    "reveived_at": 1694369937.215302
}

# candle_event_update = make_candle_event(msg)

# msg["type"] = "add"

# candle_event_add = make_candle_event(msg)
cc = Collector(exchange="kucoin", markets=["spot"])
publisher_addr = cc.PUBLISHER_ADDR
config = OhlcvRegistry
topic = "ohlcv_binance_spot_ETH/USDT_1m"

ctx = zmq.asyncio.Context()


# ======================================================================================
# callback function for testing
async def callback(update):
    logger.debug("Received OHLCV update ...")


# mock publisher for testing
async def mock_publisher(delay: float = 1):
    logger.debug("starting mock publisher...")

    publisher = ctx.socket(zmq.XPUB)
    publisher.bind(publisher_addr)

    logger.debug("OK")

    # msg = {"one": 'this', "two": 'is', "three": 'a', "four": 'test'}

    await asyncio.sleep(delay)

    try:
        for _ in range(5):
            publisher.send_multipart(
                [
                    topic.encode(),
                    json.dumps(msg, sort_keys=True, indent=4).encode(),
                ]
            )
            logger.debug("message sent: ...")
            await asyncio.sleep(0.5)

        await asyncio.sleep(2)

        publisher.send_multipart(
            [
                b"stop_it",
                json.dumps({}).encode(),
            ]
        )

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    logger.info("mock publisher stopped: OK")
    publisher.close()


async def mock_subscriber():
    logger.debug("starting mock subscriber...")

    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(OhlcvRegistry.PUBLISHER_ADDR)
    subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode())

    while True:
        try:
            msg = await subscriber.recv_multipart()
            await callback(msg)

        except (KeyboardInterrupt, asyncio.CancelledError):
            subscriber.setsockopt(zmq.UNSUBSCRIBE, topic.encode())
            break

    logger.info("mock subscriber stopped: OK")
    subscriber.close(1)


# --------------------------------------------------------------------------------------
async def test_get_initial_data() -> None:
    topic = "ohlcv_binance_spot_BTC/USDT_1m"
    repo_task = asyncio.create_task(ohlcv_repository(ctx, repo_addr))

    logger.debug("==================================================================")
    logger.debug("test_get_initial_data()")

    data = await get_initial_data(topic, ctx, repo_addr)

    if data is not None:
        logger.debug("got %s datapoints for  %s", len(data[0]), topic)
    else:
        logger.error("got empty response for %s", topic)

    repo_task.cancel()
    await asyncio.sleep(1)


# async def test_ohlcv_container_update(container: Container, runs: int) -> None:
#     for _ in range(runs):
#         await container.update(candle_event_update)


# async def test_ohlcv_container_add(container: Container, runs: int) -> None:
#     for _ in range(runs):
#         await container.update(candle_event_add)


async def test_stream_updates():
    tasks = [
        asyncio.create_task(mock_publisher(6)),
        asyncio.create_task(mock_subscriber()),
        asyncio.create_task(craeft_pond(ctx, True)),
    ]

    try:
        await asyncio.wait(tasks, timeout=18)
    except asyncio.TimeoutError:
        logger.debug("Timeout reached... exiting...")


# ==================================================================================== #
#                                       MAIN                                           #
# ==================================================================================== #
if __name__ == "__main__":
    try:
        asyncio.run(test_stream_updates())
    except KeyboardInterrupt:
        logger.info("shutting down ...")

    # ==================================================================================
    sys.exit()
    logger.setLevel(logging.ERROR)

    runs = 1
    runs_internal = 1_000_000
    st = time.time()

    for i in range(runs):
        pass

    # with Profile(timeunit=0.001) as p:
    #     for i in range(runs):
    #         asyncio.run(test_container_registry_watch())

    # (
    #     Stats(p)
    #     .strip_dirs()
    #     .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
    #     # .reverse_order()
    #     .print_stats(30)

    # )

    exc_time = (time.time() - st) * 1_000_000 / (runs * runs_internal)

    print(f"execution time: {exc_time:.2f} microseconds")
