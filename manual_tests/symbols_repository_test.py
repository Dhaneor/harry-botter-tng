#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 25 16:15:20 2024

@author dhaneor
"""

import asyncio
import random
import time
import zmq
import zmq.asyncio
from pprint import pprint

from data import markets_repository as repo  # noqa E402
from data.data_models import Symbols, Markets
from util.logger_setup import get_logger
from util.timeops import seconds_to

logger = get_logger(level="DEBUG")

ctx = zmq.asyncio.Context()

server_addr = "inproc://symbols"
client_addr = "inproc://symbols"

symbols = ("BTC/USDT", "ETH/BTC", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "ETH/GBP")


# --------------------------------------------------------------------------------------
async def get_random_exchange():
    candidates = tuple(
        [
            "alpaca",
            # "binance",
            "binanceus",
            "bitfinex",
            "bitmex",  # 'coinbasepro', 'bitfinex',
            "bitstamp",
            "bybit",
            "huobi",
            "kraken",
            "kucoin",
            "kucoinfutures",
            "okcoin",
            "okex",
            "poloniex",
            "bitrue",
        ]
    )

    return random.choice(candidates)


async def example_client(runs=3):
    socket = ctx.socket(zmq.REQ)
    socket.connect(client_addr)
    response = None

    counter, response_times, fails, errors = 0, [], 0, []

    await asyncio.sleep(2)

    while counter <= runs:
        if counter > runs-2:
            req = {'exchange': 'kucoin', 'data_type': 'markets'}
        else:
            req = {"exchange": await get_random_exchange()}

        snd_time = time.time()
        await socket.send_json(req)

        if req.get('data_type') == 'markets':
            response = Markets.from_json(await socket.recv_string())
        else:
            response = Symbols.from_json(await socket.recv_string())
        recv_time = time.time()
        response_times.append(recv_time - snd_time)

        if not response.success:
            fails += 1
            errors.append(response.errors)

        logger.info(response)
        logger.debug(response.to_dict().keys())

        logger.info("===============================================================\n")

        counter += 1
        await asyncio.sleep(0.05)

    await socket.send_json({"action": "close"})
    _ = await socket.recv_string()
    socket.close(1)

    if response and isinstance(response, Markets):
        btc = response.get("BTC/USDT", {})
        # btc['info'] = None  # too long for printing
        pprint(btc)

    logger.info(
        "average response time: %s",
        seconds_to(sum(response_times) / len(response_times)),
    )
    logger.info("failed requests: %s (%s)", fails, round(fails / runs * 100, 2))
    if errors:
        for error in errors:
            logger.error(error)

    logger.info("client: bye")
    return


async def main():
    tasks = [
        repo.markets_repository(ctx, server_addr),
        example_client()
        ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("tasks cancelled: OK")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ctx.term()

    logger.info("shutdown complete: OK")
