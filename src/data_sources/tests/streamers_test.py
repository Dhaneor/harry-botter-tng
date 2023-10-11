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
from random import choice

sys.path.append(dir(dir(__file__)))

from streamers import streamer, get_stream_manager, VALID_EXCHANGES  # noqa: E402, F401
from util.enums import SubscriptionType, MarketType  # noqa: E402, F401
from util.subscription_request import SubscriptionRequest  # noqa: E402, F401


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# --------------------------------------------------------------------------------------
async def test_streamer():
    pass


async def test_stream_manager():
    sm = get_stream_manager()
    exchange_names = list(VALID_EXCHANGES[MarketType.SPOT].keys())

    for _ in range(50):

        action = choice([True, True, True, False])
        symbol = choice(["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "ETH/BTC"])

        await sm(
            action,
            SubscriptionRequest(
                choice(exchange_names),
                MarketType.SPOT,
                SubscriptionType.TRADES,
                symbol
            )
        )

        await asyncio.sleep(7)

    await sm(b"", None)


# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    asyncio.run(test_stream_manager())
