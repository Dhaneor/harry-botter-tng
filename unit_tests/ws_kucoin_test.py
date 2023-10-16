#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 15:05:23 2021

@author_ dhaneor
"""
import asyncio
import logging
import os
import queue
import sys
import time
from cProfile import Profile
from logging.handlers import QueueHandler, QueueListener
from pstats import SortKey, Stats
from random import choice

que = queue.Queue(-1)  # no limit on size
queue_handler = QueueHandler(que)
handler = logging.StreamHandler()
listener = QueueListener(que, handler)
root = logging.getLogger("main")
root.setLevel(logging.DEBUG)
root.addHandler(queue_handler)
formatter = logging.Formatter("%(threadName)s: %(message)s")
handler.setFormatter(formatter)
listener.start()

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from config import CREDENTIALS  # noqa: E402. F401
from data_sources.websockets.i_websockets import (  # noqa: E402. F401;
    CandlesEvent
)
from data_sources.websockets.kucoin.ws_kucoin import (  # noqa: E402. F401;
    KucoinWebsocketPrivate, KucoinWebsocketPublic)

symbols = [
    "BTC-USDT",
    "ETH-USDT",
    "ADA-USDT",
    "XRP-USDT",
    "XLM-USDT",
    "abc-USDT",
    "UNI-USDT",
    "QNT-USDT",
    "ETH-BTC",
    "XRP-BTC",
    "XLM-BTC",
    "QNT-BTC",
    "KCS-BTC",
    "XRP-ETH",
    "XDC-USDT",
    "ALBT-USDT",
    "FET-USDT",
    "AAVE-USDT",
]

candles_message = {
    'data': {
        'candles': [
            '1669652580',
            '16063.1',
            '16058.2',
            '16063.2',
            '16056.7',
            '2.90872302',
            '46711.615964094'
        ],
        'symbol': 'BTC-USDT',
        'time': 1669652610226195133
    },
    'subject': 'trade.candles.update',
    'topic': '/market/candles:BTC-USDT_1min',
    'type': 'message'
}


# =============================================================================
async def test_publish(data):
    print(data)


async def tickers():
    ws = KucoinWebsocketPublic(callback=test_publish)

    try:
        # await ws.watch_ticker(symbols=[choice(symbols) for _ in range(3)])

        await ws.watch_ticker()

        while True:
            await asyncio.sleep(15)
            # await ws.unwatch_ticker(choice(symbols))
            await asyncio.sleep(15)
            # await ws.watch_ticker(choice(symbols))

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def candles():
    ws = KucoinWebsocketPublic()
    interval = "1min"

    try:
        # await ws.watch_candles(symbols=['BTC-USDT'], interval='1min')

        await ws.watch_candles(
            symbols=[choice(symbols) for _ in range(3)], interval=interval
        )

        while True:
            await asyncio.sleep(15)
            await ws.unwatch_candles(choice(symbols), interval)
            await asyncio.sleep(15)
            await ws.watch_candles(choice(symbols), interval)

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def snapshots():
    ws = KucoinWebsocketPublic()

    try:
        await ws.watch_snapshot(symbols=[choice(symbols) for _ in range(5)])

        while True:
            await asyncio.sleep(15)
            await ws.unwatch_snapshot(choice(symbols))
            await asyncio.sleep(15)
            await ws.watch_snapshot(choice(symbols))

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def balance():
    ws = KucoinWebsocketPrivate(credentials=CREDENTIALS)

    try:
        await ws.watch_balance()

        while True:
            await asyncio.sleep(15)

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def orders():
    ws = KucoinWebsocketPrivate(credentials=CREDENTIALS)

    try:
        await ws.watch_orders()

        while True:
            await asyncio.sleep(15)

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def debt_ratio():
    ws = KucoinWebsocketPrivate(credentials=CREDENTIALS)

    try:
        await ws.watch_debt_ratio()

        while True:
            await asyncio.sleep(15)

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


async def account():
    ws = KucoinWebsocketPrivate(credentials=CREDENTIALS)

    try:
        await ws.watch_account()

        while True:
            await asyncio.sleep(15)

    except KeyboardInterrupt:
        asyncio.gather()
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()


def test_candles_event(msg):
    return CandlesEvent(
        symbol=msg["data"]["symbol"],
        timestamp=msg["data"]["time"] / 1_000_000_000,
        timestamp_recv=time.time(),
        type=msg["subject"].split(".")[-1],
        interval=msg["topic"].split(":")[1].split("_")[1][:-2],
        open_time=msg["data"]["candles"][0],
        open=msg["data"]["candles"][1],
        high=msg["data"]["candles"][3],
        low=msg["data"]["candles"][4],
        close=msg["data"]["candles"][2],
        volume=msg["data"]["candles"][5],
        quote_volume=msg["data"]["candles"][6]
    )


# =============================================================================
if __name__ == "__main__":
    # asyncio.run(candles())

    x = test_candles_event(candles_message)
    print(x)

    # sys.exit()

    runs = 1_000_000
    st = time.time()

    with Profile(timeunit=0.001) as p:
        for i in range(runs):
            _ = test_candles_event(candles_message)

    (
        Stats(p)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)  # (SortKey.CALLS)
        # .reverse_order()
        .print_stats(30)

    )

    print(f"execution time: {((time.time() - st)*1_000_000/runs):.2f} microseconds")
