#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 20:15:20 2023

@author dhaneor
"""
import asyncio
import logging
import os
import random
import sys
import time
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

from src.rawi import ohlcv_repository as repo  # noqa E402

ctx = zmq.asyncio.Context()

server_addr = "inproc://ohlcv"
client_addr = "inproc://ohlcv"

symbols = ('BTC/USDT', 'ETH/BTC', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'ETH/GBP')
intervals = ('1m', '15m', '30m', '1h', '2h', '4h', '12h', '1d',)


# --------------------------------------------------------------------------------------
async def get_random_exchange():

    candidates = tuple([
        'alpaca', 'binance', 'binanceus', 'bitmex',  # 'coinbasepro', 'bitfinex',
        'bitstamp', 'bittrex', 'bybit', 'huobi', 'kraken', 'kucoin', 'kucoinfutures',
        'okcoin', 'okex', 'poloniex', 'bitrue'
    ])

    return random.choice(candidates)


async def example_client(runs=3):

    socket = ctx.socket(zmq.REQ)
    socket.connect(client_addr)

    counter, response_times, fails = 0, [], 0

    await asyncio.sleep(2)

    while counter <= runs:
        if counter > runs - 2:
            req = {
                'exchange': 'binance',  # await get_random_exchange(),
                'symbol': symbols[0],
                'interval': intervals[-1]
            }
        else:
            req = {
                'exchange': 'binance',  # await get_random_exchange(),
                'symbol': random.choice(symbols),
                'interval': "1d",  # random.choice(intervals)
            }

        snd_time = time.time()
        await socket.send_json(req)

        response = repo.Response.from_json(await socket.recv_string())
        recv_time = time.time()
        response_times.append(recv_time - snd_time)

        fails += 1 if not response.success else 0

        logger.info(response)
        if isinstance(response.to_dict(), dict):
            logger.debug(response.to_dict().keys())

        logger.info(isinstance(response.to_dict(), dict))
        logger.info("===============================================================\n")

        counter += 1
        await asyncio.sleep(0.05)

    await socket.send_json({'action': 'close'})
    _ = await socket.recv_string()
    socket.close(1)

    logger.setLevel(logging.INFO)

    logger.info(
        "average response time: %s milliseconds",
        round(sum(1000 * response_times) / len(response_times), 3)
        )
    logger.info("failed requests: %s (%s)", fails, round(fails / runs * 100, 2))
    logger.info("client: bye")
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
