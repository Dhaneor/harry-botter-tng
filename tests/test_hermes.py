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

from data import hermes
from data.data_models import Ohlcv, Markets, Symbols  # noqa: E402
from util import get_logger

logger = get_logger(level=logging.DEBUG)

hermes.USE_DB = False


# ====================================================================================
async def test_result_is_instance_of_ohlcv(repo):
    req = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "interval": "12h",
        "start": 1601753600000,
        "end": 1700760400000
    }

    response = await repo.ohlcv.get_ohlcv(**req)
    assert isinstance(response, Ohlcv)


async def test_result_is_instance_of_markets(repo):
    response = await repo.markets.all('binance')
    assert isinstance(response, Markets)
    logger.debug(f"Got {len(response.data)} markets from binance")


async def main():
    async with hermes.Hermes() as repo:
        tasks = [
            test_result_is_instance_of_ohlcv(repo),
            test_result_is_instance_of_markets(repo),
        ]
        await asyncio.gather(*tasks)

    logger.debug("context manager exited successfully")
    await asyncio.sleep(3)

# ====================================================================================
if __name__ == "__main__":
    asyncio.run(main())
