"""
Created on Sep 10 20:15:20 2023

@author dhaneor
"""
import asyncio

from data import ohlcv_repository as repo
from util import get_logger

logger = get_logger(level="DEBUG")


async def test_cache_behavior() -> None:
    req = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'interval': '1d',
        'start': '2023-12-01 00:00:00 UTC',
        'end': '2024-11-30 00:00:00 UTC',
    }

    tasks = [
        repo.process_request(req),
        repo.process_request(req)
    ]

    results = await asyncio.gather(*tasks)
    await repo.exchange_factory(None)

    assert isinstance(results[0], repo.Ohlcv)
    assert isinstance(results[1], repo.Ohlcv)

if __name__ == "__main__":
    asyncio.run(test_cache_behavior())
