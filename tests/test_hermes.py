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
import random

from data import hermes
from data.data_models import Ohlcv, Markets, Symbols
from util import get_logger

logger = get_logger(level=logging.INFO)

hermes.USE_DB = False


# ====================================================================================
async def test_result_is_instance_of_ohlcv(repo, symbol, interval, exchange):
    req = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "start": "3 years ago UTC",  # "2021-12-27 00:00:00",  # 1501753600000,
        "end": "now UTC",  # 1700760400000
    }

    response = await repo.ohlcv(**req)
    assert isinstance(response, Ohlcv)
    # assert response.start == 1501753600000
    # assert response.end == 1700760400000
    logger.debug(response)


async def test_result_is_instance_of_markets(repo):
    response = await repo.markets('binance')
    assert isinstance(response, Markets)
    logger.info(f"Got {len(response.data)} markets from binance")


async def test_result_is_instance_of_symbols(repo):
    exchange = "binance"
    response = await repo.symbols(exchange)
    assert isinstance(response, Symbols)
    logger.info(f"Got {len(response.data)} symbol names for: {exchange}")


async def main():
    interval = "1d"

    def e():
        # return 'kucoin'
        return random.choice(["binance", "bitmex", "kraken", "kucoin"])

    async with hermes.Hermes() as repo:
        tasks = [
            test_result_is_instance_of_ohlcv(repo, "BTC/USDT", interval, e()),
            test_result_is_instance_of_ohlcv(repo, "ETH/BTC", interval, e()),
            test_result_is_instance_of_ohlcv(repo, "SOL/USDT", interval, e()),
            test_result_is_instance_of_ohlcv(repo, "BNB/USDT", interval, e()),
            test_result_is_instance_of_markets(repo),
            test_result_is_instance_of_symbols(repo)
        ]
        await asyncio.gather(*tasks)

        # tasks2 = [
        #     test_result_is_instance_of_ohlcv(repo, "BTC/USDT", interval, e()),
        #     test_result_is_instance_of_ohlcv(repo, "ETH/BTC", interval, e()),
        #     test_result_is_instance_of_ohlcv(repo, "SOL/USDT", interval, e()),
        #     test_result_is_instance_of_ohlcv(repo, "BNB/USDT", interval, e()),
        #     test_result_is_instance_of_markets(repo),
        #     test_result_is_instance_of_symbols(repo)
        # ]
        # await asyncio.gather(*tasks2)

    logger.debug("context manager exited successfully")

# ====================================================================================
if __name__ == "__main__":
    asyncio.run(main())
