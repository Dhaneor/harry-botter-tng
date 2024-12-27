#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 20 16:33:20 2024

@author dhaneor
"""
import asyncio
from binance import AsyncClient
from datetime import datetime, UTC
from math import ceil

from util.logger_setup import get_logger
from data.util.convert_binance_market import convert_market

if __name__ == "__main__":
    from binance_async_ohlcv import BinanceClient
    logger = get_logger(name="main", level="DEBUG")
else:
    from .binance_async_ohlcv import BinanceClient
    logger = get_logger(f"main.{__name__}")

client = BinanceClient()

KLINES_LIMIT = 1500


class Binance:
    def __init__(self):
        self.name = 'binance'
        self.client = AsyncClient()
        self.timeframes = [
            "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",  "6h", "8h",
            "12h", "1d", "3d", "1w", "1M"
        ]
        self._info = None
        self._markets = {}
        self._symbols = []

    @property
    def symbols(self):
        if not self._info:
            raise ValueError("Exchange info not loaded. Call load_markets() first.")
        return self._symbols

    @property
    def markets(self):
        if not self._info:
            raise ValueError("Exchange info not loaded. Call load_markets() first.")
        return self._markets

    async def close(self):
        if self.client:
            await self.client.close_connection()

    async def load_markets(self):
        logger.debug("Loading markets...")
        if not self._info:
            self._info = await self.client.get_exchange_info()

        self._markets = {
            f"{market['baseAsset']}/{market['quoteAsset']}": convert_market(market)
            for market in self._info.get('symbols')
        }

        logger.debug(list(self._markets.keys())[:10])
        self._symbols = list(self._markets.keys())

    async def fetch_ohlcv(self, symbol, interval, limit=1000):
        if limit <= 1000:
            return await self.client.get_klines(
                symbol=symbol, interval=interval, limit=limit
            )
        else:
            # Calculate the number of parallel calls needed
            num_calls = ceil(limit / 1000)
            tasks = []

            # Calculate the end time (now) and start time
            end_time = int(datetime.now(UTC).timestamp() * 1000)  # in milliseconds

            # Calculate interval in milliseconds
            interval_ms = self._interval_to_milliseconds(interval)
            total_duration_ms = interval_ms * limit

            start_time = end_time - total_duration_ms

            for i in range(num_calls):
                # Calculate start and end for each batch
                batch_end = end_time - (i * 1000 * interval_ms)
                batch_start = max(start_time, batch_end - (1000 * interval_ms) + 1)

                task = asyncio.create_task(self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=KLINES_LIMIT,
                    startTime=int(batch_start),
                    endTime=int(batch_end)
                ))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Flatten the results and trim to the requested limit
            flattened = [candle for batch in reversed(results) for candle in batch]
            return flattened[:limit]

    async def fetch_status(self):
        return await self.client.get_system_status()

    async def fetch_time(self):
        return await self.client.get_server_time()

    async def fetch_ohlcv_for_period(self, symbol, interval, start, end):
        return await client.get_ohlcv(symbol, interval, start, end)

    def _interval_to_milliseconds(self, interval):
        """Convert a Binance interval string to milliseconds"""
        unit = interval[-1]

        match unit:
            case 's':
                return int(interval[:-1]) * 1000
            case 'm':
                return int(interval[:-1]) * 60 * 1000
            case 'h':
                return int(interval[:-1]) * 60 * 60 * 1000
            case 'd':
                return int(interval[:-1]) * 24 * 60 * 60 * 1000
            case 'w':
                return int(interval[:-1]) * 7 * 24 * 60 * 60 * 1000
            case 'M':
                return int(interval[:-1]) * 30 * 24 * 60 * 60 * 1000
            case _:
                raise ValueError(f"Invalid interval: {interval}")


async def main():
    binance = Binance()

    try:
        # ohlcv = await binance.fetch_ohlcv("BTCUSDT", "1h", 5000)
        # print(f"OHLCV Data - got {len(ohlcv) if ohlcv else 'NO'} candles")

        # # Test with different intervals
        # ohlcv_15m = await binance.fetch_ohlcv("BTCUSDT", "15m", 15876)
        # print(f"15m OHLCV Data - got {len(ohlcv_15m) if ohlcv_15m else 'NO'} candles")

        # ohlcv_1d = await binance.fetch_ohlcv("BTCUSDT", "1d", 500)
        # print(f"1d OHLCV Data - got {len(ohlcv_1d) if ohlcv_1d else 'NO'} candles")

        # status = await binance.fetch_status()
        # time = await binance.fetch_time()

        # print(f"Server Status: {status}")
        # print(f"Server Time: {time}")

        # time = time.get('serverTime')

        # ohlcv = await binance.fetch_ohlcv_for_period(
        #     "BTCUSDT",
        #     "1h",
        #     time - 30 * 24 * 3600 * 1000,
        #     time,
        #     )
        # print(f"OHLCV Data for Period - got {len(ohlcv) if ohlcv else 'NO'} candles")

        await binance.load_markets()
        info = binance._info

        logger.info(f"Market Info: {list(info.keys())}")
        logger.info(
            "timezone: %s, server time: %s",
            info.get('timezone'), info.get('serverTime')
            )

        for elem in info.get('rateLimits'):
            logger.info("%s", elem)
        logger.info("Exchange filters: %s", info.get('exchangeFilters'))
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        await binance.close()

if __name__ == "__main__":
    asyncio.run(main())
