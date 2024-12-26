#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a factory function that returns a CCXT exchange instance.

Use with async code, not made for synchronous code.

NOTE: For reasons of speed and efficiency, the instance for Binance is
a custom implementation which only supports the following methods:

• fetch_ohlcv
• load_markets (for symbol information)

The factory supports all exchanges from CCXT. Having this a separate
function allows for more efficient management of exchange instances
because different clients can use the same instance. The instantiation
of an instance can take some time, so it is better to do this only once.

The returned exchange instances are cached and reused if another client
requests the same exchange.

Created on December 21 14:12:20 2024

@author dhaneor
"""
import asyncio
import logging
import ccxt.pro as ccxt
from ccxt.base.errors import AuthenticationError
from uuid import uuid4

from data.rawi.util.binance_async import Binance
from util import SingletonMeta

logger = logging.getLogger(f"main.{__name__}")

RATE_LIMIT = True


def exchange_factory_fn():
    exchange_instances = {}
    busy_initializing = set()

    async def get_exchange(exchange_name: str | None = None) -> object | None:
        """Get a working exchange instance

        Call it with None (or without a parameter) to close
        all current exchange instances, which is required by CCXT
        before quitting.

        Parameters
        ----------
        exchange_name : str, Optional
            Name of the exchange to get an instance of, by default None

        Returns
        -------
        object | None
            An exchange instance, or None if the exchange does not exist
        """
        nonlocal exchange_instances
        nonlocal busy_initializing

        if exchange_name:
            exchange_name = exchange_name.lower()

        # Close all exchanges request
        if exchange_name is None:
            if not exchange_instances:
                return None

            logger.debug("Closing all exchanges ...")
            # Close all exchanges
            close_tasks = []
            for name, instance in exchange_instances.items():
                close_tasks.append(instance.close())
                logger.info(f"Closing exchange: {name}")

            if close_tasks:
                await asyncio.gather(*close_tasks)

            exchange_instances.clear()
            return None
        # if exchange_name is None:
        #     if not exchange_instances:
        #         return None

        #     for name, instance in exchange_instances.items():
        #         await instance.close()
        #         logger.info(f"Exchange closed: {name}")
        #     exchange_instances.clear()
        #     return None

        logger.debug("cached exchanges: %s" % list(exchange_instances.keys()))
        logger.debug("in preparation: %s" % list(busy_initializing))

        # to prevent two instantiations for the same exchange (because the
        # first one is still busy with loading the markets and not added to
        # the cached instances yet), we need to check if it's already in
        # preparation
        while exchange_name in busy_initializing:
            logger.debug(f"Waiting for exchange preparation: {exchange_name}")
            await asyncio.sleep(0.1)

        # Return cached exchange if it exists
        if exchange_name in exchange_instances:
            logger.debug(f"Returning cached exchange for: {exchange_name}")
            return exchange_instances[exchange_name]

        # Create a new exchange instance
        logger.info(f"Instantiating exchange: {exchange_name}")

        # some special treatment for Binance here
        if exchange_name.lower() == "binance":
            busy_initializing.add(exchange_name)
            logger.debug("added exchange to set: %s" % busy_initializing)

            exchange = Binance()
            await exchange.load_markets()

            exchange_instances[exchange_name] = exchange
            busy_initializing.remove(exchange_name)
            return exchange

        try:
            busy_initializing.add(exchange_name)
            logger.debug("added exchange to set: %s" % busy_initializing)

            exchange = getattr(ccxt, exchange_name)({"enableRateLimit": RATE_LIMIT})
            await exchange.load_markets()

            exchange_instances[exchange_name] = exchange
            busy_initializing.remove(exchange_name)
            return exchange

        except AttributeError as e:
            logger.error(f"Exchange {exchange_name.upper()} does not exist ({e})")
            return None
        except AuthenticationError as e:
            logger.error(
                "Authentication required for exchange %s (%s)",
                {exchange_name.upper()}, str(e)
                )
            return None
        except Exception as e:
            logger.error(
                f"Failed to instantiate exchange {exchange_name.upper()} ({e})"
                )
            return None

    async def close_all():
        await get_exchange(None)

    get_exchange.close_all = close_all

    return get_exchange


class ExchangeFactory(metaclass=SingletonMeta):
    def __init__(self):
        self.factory = exchange_factory_fn()
        self.id = str(uuid4())[-4:]
        logger.debug(f"ExchangeFactory instantiated with ID: {self.id}")

    def __call__(self, exchange: str) -> ccxt.Exchange:
        logger.debug("[%s] Getting exchange: %s" % (self.id, exchange))
        return self.factory(exchange)

    async def get_exchange(self, exchange: str | None = None) -> object | None:
        logger.debug("[%s] Getting exchange: %s" % (self.id, exchange))
        return await self.factory(exchange)

    async def close_all(self):
        await self.factory.close_all()


if __name__ == "__main__":
    exchange_factory = ExchangeFactory()
    exchange = asyncio.run(exchange_factory.get_exchange("binance"))
    print(exchange)
    asyncio.run(exchange_factory.close_all())
