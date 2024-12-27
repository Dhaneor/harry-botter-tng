#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a specialized symbols repository for all exchanges.

Created on Wed Dec 25 16:28:23 2024

@author_ dhaneor
"""

import asyncio

import functools
import json
import logging
import time
import zmq
import zmq.asyncio

from ccxt.base.errors import (
    NetworkError,
    ExchangeNotAvailable,
    RequestTimeout,
)
from typing import Optional

from .exchange_factory import ExchangeFactory
from .data_models import Symbols, Markets

logger = logging.getLogger("main.symbols_repository")

REQUEST_SOCKET_ADDRESS = "inproc://symbols_repository"  # change this, if necessary
KLINES_LIMIT = 1000  # number of candles to download in one call
CACHE_TTL_SECONDS = 30  # cache TTL in seconds

LOG_STATUS = False  # enable logging of server status and server time

MAX_RETRIES = 3  # maximum number of retries for fetching OHLCV data
RETRY_DELAY = 5.0  # delay between retries in seconds

exchange_factory = ExchangeFactory()


# ====================================================================================
def cache(ttl_seconds: int = CACHE_TTL_SECONDS):
    """
    Decorator function to cache OHLCV (Open, High, Low, Close, Volume) data.

    This decorator implements a caching mechanism for OHLCV data. It stores the
    results of the decorated function in a dictionary, using a tuple of
    (exchange, symbol, interval) as the key. If the same request is made again,
    it returns the cached data instead of calling the original function.

    Parameters:
    func (callable): The function to be decorated. It should be an asynchronous
                     function that takes an Symbols object as an argument and
                     returns an Symbols object.

    Returns:
    callable: A wrapper function that implements the caching logic.

    The wrapper function:
    - Takes a Symbols object as an argument.
    - Returns a Symbols object, either from the cache or by calling the
      original function.
    """

    def decorator(func):
        symbols_cache: dict[tuple[str, str, str], tuple[dict, float]] = {}

        @functools.wraps(func)
        async def wrapper(response, exchange_factory) -> Symbols:
            start_time = time.time()
            current_time = start_time
            data_type = response.__class__.__name__.lower()

            # Clean expired cache entries
            expired_keys = [
                key
                for key, (_, timestamp) in symbols_cache.items()
                if current_time - timestamp > ttl_seconds
            ]
            for key in expired_keys:
                del symbols_cache[key]

            cache_key = (response.exchange)

            # return cached data if available and not expired
            if cache_key in symbols_cache:
                cached_data, timestamp = symbols_cache[cache_key]
                if current_time - timestamp <= ttl_seconds:
                    logger.debug(
                        "Returning cached %s data for %s", data_type, cache_key,
                        )
                    response.data = cached_data
            # fetch data from exchange if not available in cache
            else:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await func(response, exchange_factory)
                        if response.data:
                            symbols_cache[cache_key] = (response.data, current_time)
                            logger.debug("Cached %s data for %s", data_type, cache_key)
                        break
                    # retry in case of network error, exchange not
                    # available, or request timeout
                    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
                        if attempt == MAX_RETRIES - 1:
                            logger.error(
                                "Max retries reached for %s: %s", cache_key, str(e)
                            )
                            response.network_error = str(e)
                        else:
                            logger.warning(
                                f"Retry {attempt + 1} for {cache_key} due to: {str(e)}"
                            )
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    # abort if other exceptions occur
                    except Exception as e:
                        logger.error(
                            "unknown error in cache funtion: %s", e, exc_info=True
                            )
                        break

            response.execution_time = time.time() - start_time
            return response
        return wrapper
    return decorator


# @cache()
async def get_symbols_data_from_exchange(response: Symbols) -> Symbols:
    # try to get a working exchange instance
    if (exchange := await exchange_factory.get_exchange(response.exchange)):
        response.data = exchange.symbols
    else:
        response.exchange_error = f"Exchange {response.exchange} not available"

    return response


# @cache()
async def get_markets_data_from_exchange(response: Markets) -> Markets:
    # try to get a working exchange instance
    if (exchange := await exchange_factory.get_exchange(response.exchange)):
        response.data = exchange.markets
    else:
        response.exchange_error = f"Exchange {response.exchange} not available"

    return response


async def get_symbols(
    req: dict,
    socket: zmq.asyncio.Socket | None = None,
    id_: bytes | None = None,
) -> None:
    """Process a client request for OHLCV data.

    Parameters
    ----------
    req: dict
        dictionary with mandatory keys:
        • "exchange", "symbol", "interval"
        and optional keys:
        • "start", "end"

    exchange_factory: Callable | None
        A factory function for creating exchange instances.
        If not provided, the default exchange_factory will be
        used, default is None

    socket: zmq.asyncio.Socket
        a working/initialized ZeroMQ socket, default is None.

    id_: bytes
        caller identity of the request, default is None
    """

    # Create a Symbols object with the request details
    response = Symbols(
        exchange=req.get("exchange", "NOT PROVIDED"),
        socket=socket,
        id=id_
        )

    logger.debug(response)

    # proceed only if a valid Symbols object has been created
    if response.success:
        response = await get_symbols_data_from_exchange(response)

    if response.socket:
        await response.send()

    return response


async def get_markets(
    req: dict,
    socket: zmq.asyncio.Socket | None = None,
    id_: bytes | None = None,
) -> None:
    """Process a client request for OHLCV data.

    Parameters
    ----------
    req: dict
        dictionary with mandatory keys:
        • "exchange", "symbol", "interval"
        and optional keys:
        • "start", "end"

    socket: zmq.asyncio.Socket
        a working/initialized ZeroMQ socket, default is None.

    id_: bytes
        caller identity of the request, default is None
    """

    # Create a Markets object with the request details
    response = Markets(
        exchange=req.get("exchange", "NOT PROVIDED"),
        socket=socket,
        id=id_
        )

    logger.debug(response)

    # proceed only if a valid Marktets object has been created
    if response.success:
        response = await get_markets_data_from_exchange(response)

    if response.socket:
        await response.send()

    return response


async def markets_repository(
    ctx: Optional[zmq.asyncio.Context] = None,
    addr: Optional[str] = None
):
    """Start the OHLCV repository.

    Clients can request OHLCV data by sending a (JSON encoded) dictionary.

    example:

    ..code-block:: python
        {"exchange": "binance", "symbol": "BTC/USDT", "interval": "1m"}

    Parameters
    ----------
    ctx : zmq.asyncio.Context, optional
        ZMQ context to use. If None, a new context is created. This
        allows for flexibility wrt how this is used.
        You can run this function in a separate process, or in a
        separate thread and provide no context, or you can combine the
        repository with other components and use the provided/shared
        context, by default None

    addr : str, optional
        Address to bind to, only necessary if you want to overwrite
        the REQUEST_SOCKET_ADDRESS (defined in the 'zmq_config' file in
        this directory), by default None
    """
    context = ctx or zmq.asyncio.Context()
    addr = addr or REQUEST_SOCKET_ADDRESS

    socket = context.socket(zmq.ROUTER)
    socket.bind(addr)

    poller = zmq.asyncio.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        try:
            events = dict(await poller.poll())

            if socket in events:
                msg = await socket.recv_multipart()
                logger.debug("received message: %s", msg)
                identity, request = msg[0], msg[2].decode()

                logger.debug("received request: %s from %s", request, identity)

                request = json.loads(request)

                if request.get("action") == "close":
                    logger.info("shutdown request received ...")
                    await exchange_factory.get_exchange(None)
                    socket.send_multipart([identity, b"", b"OK"])
                    break

                # process request in the background
                if request.get('data_type') == 'markets':
                    asyncio.create_task(
                        get_markets(request, exchange_factory, socket, identity)
                        )
                else:
                    asyncio.create_task(
                        get_symbols(request, exchange_factory, socket, identity)
                        )

        except asyncio.CancelledError:
            logger.info("task cancelled -> closing exchange ...")
            break
        except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info("task cancelled -> closing exchange ...")

            await socket.send_json([])
            break

    # cleanup
    await exchange_factory.get_exchange(None)
    socket.close(1)

    logger.info("symbols repository shutdown complete: OK")


if __name__ == "__main__":
    ctx = zmq.asyncio.Context()

    try:
        asyncio.run(markets_repository(ctx))
    except KeyboardInterrupt:
        ctx.term()
