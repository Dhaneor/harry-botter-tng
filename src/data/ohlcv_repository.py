#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a specialized OHLCV repository for the OHLCV containers.

classes
    Ohlcv

    This class standardizes the response, does type checks for the
    values from the request, and includes a method for sending the
    response after fetching the OHLCV data (including some error
    flags for the client).

functions
    ohlcv_repository

    This is the only function that needs to be used/called to start the
    repository which can be accessed by clients via its ZeroMQ socket.
    All other functions in this module support this main function.

    Parameters
        ctx Optional[zmq.asyncio.Context]
            a ZeroMQ Context object, only necessary if multiple ZerOMQ
            connections/functions are running in the same thread;
            otherwise THE Context will be initialized and managed here
        addr: Optional[str]
            the 0MQ address to connect to

Created on Sat Sep 16 13:58:23 2023

@author_ dhaneor
"""

import asyncio

import ccxt.pro as ccxt
import functools
import json
import logging
import time
import zmq
import zmq.asyncio

from ccxt.base.errors import (
    BadSymbol,
    BadRequest,
    AuthenticationError,
    InsufficientFunds,
    NetworkError,
    ExchangeNotAvailable,
    RequestTimeout,
    ExchangeError,
)
from datetime import datetime

from .exchange_factory import ExchangeFactory
from .data_models import Ohlcv
from .util.timestamp_converter import timestamp_converter
from util import unix_to_utc

logger = logging.getLogger("main.ohlcv_repository")

REQUEST_SOCKET_ADDRESS = "inproc://ohlcv_repository"  # change this, if necessary
KLINES_LIMIT = 1000  # number of candles to download in one call
CACHE_TTL_SECONDS = 30  # cache TTL in seconds

LOG_STATUS = False  # enable logging of server status and server time

MAX_RETRIES = 3  # maximum number of retries for fetching OHLCV data
RETRY_DELAY = 5.0  # delay between retries in seconds

exchange_factory = ExchangeFactory()


# ====================================================================================
async def log_server_time(exchange) -> None:
    """Get the server time from the exchange."""
    try:
        if exchange.name.lower() == "binance":
            time = await exchange.fetch_time()
            time = datetime.fromtimestamp(time.get("serverTime") / 1000).isoformat()
            time = time.split("T")[1][:-7]
        else:
            time = exchange.iso8601(await exchange.fetch_time()).split("T")[1][:-5]
    except Exception as e:
        logger.error(f"Error fetching server time from exchange: {e}")
        time = None

    if time:
        logger.info("Server time: %s" % time)


async def log_server_status(exchange) -> None:
    """Log the server status from the exchange."""
    try:
        status = await exchange.fetch_status()
        status = "OK" if status["status"] == "ok" else status
        logger.info("Server status: %s" % status)
    except Exception as e:
        logger.error(f"Error fetching server status from exchange: {e}")


def cache_ohlcv(ttl_seconds: int = CACHE_TTL_SECONDS):
    """
    Decorator function to cache OHLCV (Open, High, Low, Close, Volume) data.

    This decorator implements a caching mechanism for OHLCV data. It stores the
    results of the decorated function in a dictionary, using a tuple of
    (exchange, symbol, interval) as the key. If the same request is made again,
    it returns the cached data instead of calling the original function.

    Parameters:
    func (callable): The function to be decorated. It should be an asynchronous
                     function that takes an Ohlcv object as an argument and
                     returns an Ohlcv object.

    Returns:
    callable: A wrapper function that implements the caching logic.

    The wrapper function:
    - Takes a Ohlcv object as an argument.
    - Returns a Ohlcv object, either from the cache or by calling the
      original function.
    """

    def decorator(func):
        ohlcv_cache: dict[tuple[str, str, str], tuple[dict, float]] = {}

        @functools.wraps(func)
        async def wrapper(response, exchange) -> Ohlcv:
            start_time = time.time()
            current_time = start_time

            # Clean expired cache entries
            expired_keys = [
                key
                for key, (_, timestamp) in ohlcv_cache.items()
                if current_time - timestamp > ttl_seconds
            ]
            for key in expired_keys:
                del ohlcv_cache[key]

            cache_key = (
                response.exchange, response.symbol, response.interval,
                response.start, response.end
                )

            if cache_key in ohlcv_cache:
                cached_data, timestamp = ohlcv_cache[cache_key]
                if current_time - timestamp <= ttl_seconds:
                    logger.debug(f"Returning cached OHLCV data for {cache_key}")
                    response.data = cached_data
                    response.cached = True
            else:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await func(response, exchange)
                        if response.data:
                            ohlcv_cache[cache_key] = (response.data, current_time)
                            logger.debug(
                                "OHLCV data for %s added to cache." % f"{cache_key}"
                                )
                        break
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

            response.execution_time = time.time() - start_time
            return response

        return wrapper

    return decorator


async def fetch_with_ccxt(
    response: Ohlcv, exchange: ccxt.Exchange, max_retries: int = 3
) -> Ohlcv:
    start_time, end_time, data = response.start, response.end, None

    interval_errors = (
        "period",
        "interval",
        "timeframe",
        "binSize",
        "candlestick",
        "step",
    )

    logger.debug(
        "Fetching OHLCV data for %s from %s to %s...",
        response.symbol, unix_to_utc(start_time), unix_to_utc(end_time)
    )

    for attempt in range(max_retries):
        try:
            data = await exchange.fetch_ohlcv(
                symbol=response.symbol,
                timeframe=response.interval,
                since=start_time,
                params={'until': end_time}
            )
            break
        except (NetworkError) as e:
            if attempt == max_retries - 1:
                logger.error(
                    "Failed to fetch OHLCV data after %s attempts: %s",
                    max_retries, str(e)
                    )
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except AuthenticationError as e:
            logger.error(f"[AuthenticationError] {str(e)}")
            response.authentication_error = str(e)
        except BadSymbol as e:
            logger.error(f"[BadSymbol] {str(e)}")
            response.symbol_error = True
        # ccxt is inconsistent when encountering an error
        # that is caused by an invalid interval. some special
        # handling is required here
        except InsufficientFunds as e:
            logger.error(f"[InsufficientFunds] {str(e)}")
            response.interval_error = True
        except BadRequest as e:
            logger.error(f"[BadRequest] {str(e)}")
            if any(s in str(e) for s in response.interval):
                response.interval_error = True
            else:
                response._bad_request_error = str(e)
        except ExchangeError as e:
            logger.error(f"[ExchangeError] {str(e)}")
            if "poloniex" in str(e):
                response.interval_error = True
            elif any(s in str(e) for s in interval_errors):
                response.interval_error = True
            else:
                response.exchange_error = str(e)
        except ExchangeNotAvailable as e:
            logger.error(f"[ExchangeNotAvailable] {str(e)}")
            if any(s in str(e) for s in interval_errors):
                response.interval_error = True
            else:
                response.exchange_error = str(e)
        except Exception as e:
            logger.error(f"[Unexpected Error] {str(e)}", exc_info=True)
            response.unexpected_error = str(e)

    # Filter and sort the data
    if data:
        response.data = sorted(
            [candle for candle in data if start_time <= candle[0] <= end_time],
            key=lambda x: x[0]
        )

    return response


async def fetch_with_binance(response: Ohlcv, exchange: ccxt.Exchange) -> Ohlcv:
    logger.debug(
        f"Fetching OHLCV data for {response.symbol} from "
        f"{response.start} to {response.end}"
        )
    result = await exchange.fetch_ohlcv_for_period(
        symbol="".join(response.symbol.split("/")),
        interval=response.interval,
        start=response.start,
        end=response.end,
    )

    response.data = [
        [float(row[i]) if i != 0 else int(row[i]) for i in range(len(row) - 1)]
        for row in result
    ]  # Convert to ccxt format (Binance returns strings)

    return response


@cache_ohlcv()
async def get_ohlcv(response: Ohlcv, exchange: ccxt.Exchange) -> Ohlcv:
    """Get OHLCV data for a given exchange, symbol and interval.

    Parameters
    ----------
    response

    Returns
    -------
    Ohlcv
    """

    if not hasattr(exchange, "fetch_ohlcv"):
        logger.error(
            "%s does not support fetching OHLCV data" % response.exchange
            )
        response.fetch_ohlcv_not_available = True
        return response

    if response.interval not in exchange.timeframes:
        logger.error(
            "Invalid interval for %s: %s" % (response.exchange, response.interval)
            )
        response.interval_error = True
        return response

    if response.symbol not in exchange.symbols:
        logger.error(
            "Invalid symbol for %s: %s" % (response.exchange, response.symbol)
            )
        response.symbol_error = True
        return response

    # ................................................................................

    # with Binance we can use parallel calls to make this step faster
    if exchange.name == "binance":
        return await fetch_with_binance(response, exchange)
    else:
        return await fetch_with_ccxt(response, exchange)


@timestamp_converter()
async def convert_dates(exchange, symbol, interval, start, end):
    """Dummy function to convert dates in the request dictionary."""
    return {
        'exchange': exchange,
        'symbol': symbol,
        'interval': interval,
        'start': start,
        'end': end,
    }


async def process_request(
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
    req = await convert_dates(**req)

    # Create a Ohlcv object with the request details
    response = Ohlcv(
        exchange=req.get("exchange"),
        symbol=req.get("symbol"),
        interval=req.get("interval"),
        start=req.get("start"),
        end=req.get("end"),
        socket=socket,
        id=id_,
    )

    logger.debug(response)

    # proceed only if a valid Ohlcv object has been created
    if response.success:
        # try to get a working exchange instance
        if not (exchange := await exchange_factory(response.exchange)):
            logger.error(f"Exchange {response.exchange} not available")
            response.exchange_error = f"Exchange {response.exchange} not available"
            if response.socket:
                await response.send()
            return response

        if LOG_STATUS:
            tasks = (
                asyncio.create_task(log_server_time(exchange)),
                asyncio.create_task(log_server_status(exchange)),
                asyncio.create_task(get_ohlcv(response=response, exchange=exchange)),
            )

            response = next(
                filter(lambda x: x is not None, await asyncio.gather(*tasks))
            )
        else:
            response = await get_ohlcv(response=response, exchange=exchange)

    if response.socket:
        await response.send()

    return response


async def ohlcv_repository(
    ctx: zmq.asyncio.Context | None = None,
    addr: str | None = None,
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
                    await exchange_factory(None)
                    socket.send_multipart([identity, b"", b"OK"])
                    break

                # process request in the background
                asyncio.create_task(process_request(request, socket, identity))

        except asyncio.CancelledError:
            logger.info("task cancelled -> closing exchange ...")
            break
        except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info("task cancelled -> closing exchange ...")

            await socket.send_json([])
            asyncio.sleep(0.001)
            break

    # cleanup
    await exchange_factory.close_all()
    await asyncio.sleep(3)
    socket.close(1)

    logger.info("ohlcv repository shutdown complete: OK")


if __name__ == "__main__":
    ctx = zmq.asyncio.Context()

    try:
        asyncio.run(ohlcv_repository(ctx))
    except KeyboardInterrupt:
        ctx.term()
