#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 20:15:20 2023

@author dhaneor
"""
import asyncio
import json
import logging
import os
import random
import sys
import zmq
import zmq.asyncio

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

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

from src.rawi import ohlcv_repository as repo  # noqa E402

ctx = zmq.asyncio.Context()

server_addr = "inproc://ohlcv"
client_addr = "inproc://ohlcv"

symbols = ('BTC/USDT', 'ETH/BTC', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'ETH/GBP')
intervals = ('1m', '15m', '1h', '5h', '4h', '12h', '1d',)


# --------------------------------------------------------------------------------------
async def get_random_exchange():

    candidates = tuple([
        'alpaca', 'binance', 'binanceus', 'bitmex',  # 'coinbasepro', 'bitfinex',
        'bitstamp', 'bittrex', 'bybit', 'huobi', 'kraken', 'kucoin', 'kucoinfutures',
        'okcoin', 'okex', 'poloniex', 'bitrue'
    ])

    return random.choice(candidates)


async def example_client(runs=10):

    socket = ctx.socket(zmq.REQ)
    socket.connect(client_addr)

    counter = 0

    await asyncio.sleep(2)

    while counter <= runs:
        if counter > runs - 2:
            req = {
                'exchange': 'binance',  # await get_random_exchange(),
                'symbol': symbols[0],
                'interval': intervals[0]
            }
        else:
            req = {
                'exchange': await get_random_exchange(),
                'symbol': "BTC/USDT",  # random.choice(symbols),
                'interval': "1h"  # random.choice(intervals)
            }

        logger.debug("sending request %s: %s", counter, req)

        await socket.send_json(req)

        response = repo.Response.from_json(await socket.recv_string())

        logger.info(response)

        # msg = json.loads(await socket.recv_string())

        # if msg and msg.get('success'):

        #     if data := msg.get('data'):
        #         logger.info("%s ...", data[-1])
        #     else:
        #         logger.warning(
        #             "No data received even though message indicates success."
        #             )
        #         logger.warning(msg)

        # elif msg and not msg.get('success'):
        #     logger.error(">>>>> >>>>> REQUEST ERRORS: %s", msg.get('errors'))

        # else:
        #     logger.error("no message received")

        logger.info("===============================================================\n")
        counter += 1

        await asyncio.sleep(1)

    await socket.send_json({'action': 'close'})
    _ = await socket.recv_string()
    socket.close(1)
    return


async def main():
    tasks = [
        repo.ohlcv_repository(ctx, server_addr),
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
