#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 20:15:20 2023

@author dhaneor
"""
import asyncio
import ccxt
import json
import logging
import os
import random
import sys
import zmq
import zmq.asyncio

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

from src.data_sources import ohlcv_repository as repo  # noqa E402

ctx = zmq.asyncio.Context()

server_addr = "inproc://ohlcv"
client_addr = "inproc://ohlcv"


# --------------------------------------------------------------------------------------
async def get_random_exchange():

    candidates = tuple([
        'alpaca', 'binance', 'binanceus', 'bitmex', 'coinbasepro', 'bitfinex',
        'bitstamp', 'bittrex', 'bybit', 'huobi', 'kraken', 'kucoin', 'kucoinfutures',
        'okcoin', 'okex', 'poloniex', 'bitrue'
    ])

    return random.choice(candidates)


async def example_client():

    socket = ctx.socket(zmq.REQ)
    socket.connect(client_addr)

    counter = 0

    await asyncio.sleep(2)

    while counter < 10:
        req = {
            'exchange': await get_random_exchange(),
            'symbol': 'XRP/USDT',
            'interval': '1h'
        }

        logger.debug("sending request %s: %s", counter, req)

        await socket.send_json(req)

        msg = json.loads(await socket.recv_string())

        if msg:
            logger.debug(msg[-1])
        else:
            logger.error("...")

        logger.debug("----------------------------------------------------------------")
        counter += 1

        await asyncio.sleep(0.1)

    await socket.send_json({'action': 'close'})
    _ = await socket.recv_json()
    socket.close()


async def main():
    tasks = [
        repo.main(ctx, server_addr),
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
