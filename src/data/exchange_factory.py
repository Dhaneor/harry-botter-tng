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
import logging
import ccxt.pro as ccxt
from ccxt.base.errors import AuthenticationError

from .rawi.util.binance_async import Binance

logger = logging.getLogger(f"main.{__name__}")

RATE_LIMIT = True


def exchange_factory_fn():
    exchange_instances = {}

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

        if exchange_name:
            exchange_name = exchange_name.lower()

        # Close all exchanges request
        if exchange_name is None:
            for name, instance in exchange_instances.items():
                await instance.close()
                logger.info(f"Exchange closed: {name}")
            exchange_instances.clear()
            return None

        # Return cached exchange if it exists
        if exchange_name in exchange_instances:
            logger.debug(f"Returning cached exchange for: {exchange_name}")
            return exchange_instances[exchange_name]

        # Create a new exchange instance
        logger.info(f"Instantiating exchange: {exchange_name}")

        # some special treatment for Binance here
        if exchange_name.lower() == "binance":
            exchange = Binance()
            await exchange.load_markets()
            exchange_instances[exchange_name] = exchange
            return exchange

        try:
            exchange = getattr(ccxt, exchange_name)({"enableRateLimit": RATE_LIMIT})
            await exchange.load_markets()
            exchange_instances[exchange_name] = exchange
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

        active = exchange_instances
    return get_exchange
