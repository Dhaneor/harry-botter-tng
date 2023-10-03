#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:44:23 2022

@author_ dhaneor
"""
import asyncio
import logging
import os
import sys
from typing import Literal, TypeAlias

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------
from src.data_sources.streamer import (  # noqa E402
    start_streaming_server,
    streamer,
    mock_streamer
)

Mode: TypeAlias = Literal[
    'tickers', 'candles', 'snapshots', 'all_tickers', 'all_snapshots'
]


# =============================================================================
async def test_streamer(mode: Mode):
    try:
        if mode == 'mock':
            task = asyncio.create_task(
                mock_streamer(ctx=None, msg_per_sec=5_000, repeat=2)
            )
        else:
            task = asyncio.create_task(
                streamer(market='kucoin.margin', mode=mode)
            )

        logger.info("waiting for task to finish ...")

        await asyncio.gather(task, return_exceptions=True)

    except asyncio.CancelledError:
        task.cancel()
        await asyncio.sleep(1)

        # logger.info("task done: %s", task.done())
        # logger.info(task.cancelled())

        await task

        # await asyncio.gather(task, return_exceptions=True)

    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt.in test_streamer')
    except Exception as e:
        logger.exception(e)
    finally:
        logger.info("cleanup complete: OK")


def test_streaming_server():
    members = ['candles', 'all_tickers', 'all_snapshots']
    # members = ['all_snapshots']
    start_streaming_server(members=members)  # type:ignore


# =============================================================================
if __name__ == '__main__':
    try:
        asyncio.run(test_streamer(mode='mock'))
    except KeyboardInterrupt:
        pass  # logger.info("keyboard interrupt caught with asyncio.run()")
    finally:
        logger.info("shut down complete: OK")

    # test_start_streaming_server()
