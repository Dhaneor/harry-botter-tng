#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the tests for the Hermes class.

Hermes is the god of wisdom and knowledge. He knows everything about:

• OHLCV data for all cryptocurrencies (on all exchanges supported by CCXT)
• Symbol information (like trading pairs, currencies, etc.)

• Sentiment (future feature)
• On-chain data (future feature)

Created on December 21 07:59:20 2024

@author dhaneor
"""
import asyncio
import logging
import os
import sys

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------
from src.data.hermes import Hermes  # noqa: E402
from src.data.rawi.ohlcv_repository import Response  # noqa: E402

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)


# ====================================================================================
async def test_result_is_instance_of_response():
    async with Hermes() as repo:
        response = await repo.get_ohlcv(
            exchange='binance',
            symbol='BTC/USDT',
            interval='1m',
            start=1671753600000,  # 2023-01-01 00:00:00
            end=1671760400000  # 2023-01-01 01:00:00
            )
    assert isinstance(response, Response)


# ====================================================================================
if __name__ == "__main__":
    asyncio.run(test_result_is_instance_of_response())
