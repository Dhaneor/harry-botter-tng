#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:39:23 2023

@author_ dhaneor
"""
import asyncio
import logging
import sys

from os.path import dirname as dir

sys.path.append(dir(dir(__file__)))

from streamers import streamer, get_stream_manager  # noqa: E402, F401
from util.enums import SubscriptionType, MarketType  # noqa: E402, F401
from util.subscription_request import SubscriptionRequest  # noqa: E402, F401


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# --------------------------------------------------------------------------------------
async def test_streamer():
    pass


async def test_stream_manager():
    workers = {}
    sm = get_stream_manager(workers)

    await sm(
        b"",
        SubscriptionRequest(
            "binance", MarketType.SPOT, SubscriptionType.TRADES, "BTC/USDT"
        )
    )

    await sm(
        b"",
        SubscriptionRequest(
            "kucoin", MarketType.SPOT, SubscriptionType.TRADES, "BTC/USDT"
        )
    )

    # await sm(
    #     b"",
    #     SubscriptionRequest(
    #         "binance", MarketType.SPOT, SubscriptionType.TRADES, "BTC/USDT"
    #     )
    # )

    logger.info("test_stream_manager started: OK")

    await asyncio.sleep(36000)

    await sm(b"", None)


# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    asyncio.run(test_stream_manager())
