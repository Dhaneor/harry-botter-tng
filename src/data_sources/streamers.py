#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a webocket streamer components.

Created on Tue Oct 10  09:10:23 2023

@author_ dhaneor
"""
import asyncio
import ccxt.pro as ccxt
import logging
import time
import zmq
import zmq.asyncio

from ccxt.base.errors import BadSymbol, NetworkError, ExchangeNotAvailable
from functools import partial
from typing import Coroutine, TypeVar, Optional

from util.sequence import sequence
from util.enums import SubscriptionType, MarketType
from util.subscription_request import SubscriptionRequest
from zmqbricks.gond import Gond
from zmq_config import Streamer, BaseConfig

logger = logging.getLogger("main.streamers")

# types
ContextT = TypeVar("ContextT", bound=zmq.asyncio.Context)
ConfigT = TypeVar("ConfigT", bound=BaseConfig)
SocketT = TypeVar("SocketT", bound=zmq.Socket)
ExchangeT = TypeVar("ExchangeT", bound=ccxt.Exchange)

SLEEP_ON_ERROR = 10  # seconds
FREQ_UPDATES = 100  # milliseconds
LIMIT_UPDATES = 1000  # max number of updates for one request

VALID_EXCHANGES = {
    MarketType.SPOT: {
        "binance": ccxt.binance,
        "bitfinex": ccxt.bitfinex,
        "bitmex": ccxt.bitmex,
        "bittrex": ccxt.bittrex,
        "bybit": ccxt.bybit,
        "kraken": ccxt.kraken,
        "kucoin": ccxt.kucoin,
    },
    MarketType.FUTURES: {
        "binance": ccxt.binanceusdm,
        "kucoin": ccxt.kucoinfutures,
    },
}


# ======================================================================================
async def get_exchange_instance(exc_name: str, market: MarketType) -> ExchangeT:
    # public API makes no  difference between spot and margin
    market = MarketType.SPOT if market == MarketType.MARGIN else market

    if exc_name in vars(ccxt) and exc_name in VALID_EXCHANGES[market].keys():
        return getattr(ccxt, exc_name)({"newUpdates": True})
    else:
        raise ExchangeNotAvailable(exc_name)


async def close_exchange(name: str, exchanges: dict) -> None:
    try:
        await exchanges[name].close()
    except KeyError as e:
        logger.warning(
            "unable to close exchange instance, not found: %s --> %s", name, e
        )
    except Exception as e:
        logger.error("unable to close exchange instance, unexpected erorr: %s", e)
    else:
        logger.info("exchange instance closed: %s", name)
        del exchanges[name]


async def close_everything(workers: dict, exchanges: dict) -> None:
    # cancel worker tasks
    logger.info("shutdown requested ...")
    tasks = []
    for exchange in workers.values():
        for sub_type in exchange.values():
            for topic in sub_type.values():
                logger.info("cancelling task for topic: %s", topic)
                topic.cancel()
                tasks.append(topic)

    await asyncio.gather(*tasks, return_exceptions=False)

    # close exchange instance(s)
    for exc_name in list(exchanges.keys()):
        await close_exchange(exc_name, exchanges)


# --------------------------------------------------------------------------------------
@sequence(logger=logger, identifier="seq")
async def send(msg: dict, socket: zmq.Socket):
    await socket.send_json(msg)


@sequence(logger=logger, identifier="seq")
async def log_message(msg: dict):
    pass
    # logger.info("---------------------------------------------------------------------")
    # logger.info("[%s] %s", msg["sequence"], msg)
    # logger.info(msg)


async def process_ohlcv(msg: list) -> dict:
    logger.info(f"Processing OHLCV: {msg}")
    return {
        "timestamp": msg[0],
        "open": msg[1],
        "high": msg[2],
        "low": msg[3],
        "close": msg[4],
        "volume": msg[5],
    }


# --------------------------------------------------------------------------------------
async def worker(
    rcv_coro: Coroutine,
    snd_coro: Coroutine,
    use_since: bool,
    add_to_result: Optional[dict] = None,
    process: Optional[Coroutine] = None,
    limit: Optional[int] = LIMIT_UPDATES,
    freq: Optional[int] = FREQ_UPDATES,
) -> None:
    logger.info(add_to_result)

    async def process_update(update: dict) -> dict:
        if "info" in update:
            del update["info"]
        update = await process(update) if process is not None else update
        return add_to_result | update if add_to_result is not None else update

    counter, since = 0, time.time()

    while True:
        try:
            if use_since:
                data = await rcv_coro(since=since, limit=limit)
            else:
                data = await rcv_coro()

        except ExchangeNotAvailable as e:
            logger.error("[%s] %s", counter, e)
            await asyncio.sleep(SLEEP_ON_ERROR)
        except BadSymbol as e:
            logger.error("[%s] %s", counter, e)
            break
        except NetworkError as e:
            logger.error("[%s] CCXT network error: %s", counter, e, exc_info=1)
            await asyncio.sleep(SLEEP_ON_ERROR)
        except asyncio.CancelledError:
            logger.info("Cancelled...")
            break
        except Exception as e:
            logger.exception(e)
            # break
        else:
            data = [data] if not isinstance(data, list) else data

            if data:
                if use_since is not None:
                    since = data[-1].get("timestamp") + 1
                    if freq:
                        await asyncio.sleep(freq / 1000)

                for update in data:
                    await snd_coro(await process_update(update))

                counter += 1


async def create_worker(
    sub_req: SubscriptionRequest,
    workers: dict,
    exchanges: dict
) -> None:
    # create exchange instance, if it doesn't exist yet
    if sub_req.exchange not in workers:
        exchanges[sub_req.exchange] = exchange = await get_exchange_instance(
            sub_req.exchange, sub_req.market
        )
        # create necessary keys in workers registry
        workers[sub_req.exchange] = {}
        logger.info("exc --> %s", exchanges[sub_req.exchange])
    else:
        exchange = exchanges[sub_req.exchange]

    # prepare parameters for worker coroutine
    match sub_req.sub_type:

        case SubscriptionType.OHLCV:
            rcv_coro = partial(
                exchange.watch_ohlcv,
                symbol=sub_req.symbol,
                timeframe=sub_req.interval,
            )
            use_since = True

        case SubscriptionType.BOOK:
            rcv_coro = partial(exchange.watch_order_book, symbol=sub_req.symbol)
            use_since = False

        case SubscriptionType.TRADES:
            rcv_coro = partial(exchange.watch_trades, symbol=sub_req.symbol)
            use_since = True

        case SubscriptionType.TICKER:
            rcv_coro = partial(exchange.watch_ticker, symbol=sub_req.symbol)
            use_since = False

        case _:
            raise ValueError(f"invalid subscription type: {sub_req.sub_type}")

    process = process_ohlcv if sub_req.sub_type == SubscriptionType.OHLCV else None
    add_to_result = {"exchange": sub_req.exchange, "market": sub_req.market}

    # create a worker task for the topic
    workers[sub_req.exchange][sub_req.market] = {}
    workers[sub_req.exchange][sub_req.market][sub_req.topic] = asyncio.create_task(
        worker(rcv_coro, log_message, use_since, add_to_result, process)
    )


async def remove_worker(
    sub_req: SubscriptionRequest,
    workers: dict,
    exchanges: dict
) -> None:
    try:
        workers[sub_req.exchange][sub_req.market][sub_req.topic].cancel()
    except KeyError as e:
        logger.warning("unable to remove worker, not found for: %s --> %s", sub_req, e)
        return
    else:
        del workers[sub_req.exchange][sub_req.market][sub_req.topic]

    if not workers[sub_req.exchange][sub_req.market]:
        del workers[sub_req.exchange][sub_req.market]

    if not workers[sub_req.exchange]:
        await close_exchange(sub_req.exchange, exchanges)
        del workers[sub_req.exchange]


def get_stream_manager():
    exchanges: dict = {}
    workers: dict = {}
    topics: dict[str, int] = {}

    async def stream_manager(
        action: bytes, sub_req: Optional[SubscriptionRequest] = None
    ) -> None:
        """Manages creation/removal of stream workers.

        Parameters
        ----------
        action : bytes
            subscribe (1) or unsubscribe (0)
        sub_req : SubscriptionRequest, optional
            (un)subscribe request, by default None
            For clean shutdown. If this is None, then we will
            cancel all tasks/saubscriptions, and close the
            exchange instances.
        """
        # log request
        action_str = "subscribe" if action else "unsubscribe"
        logger.info("received %s request --> %s", action_str.upper(), sub_req)

        # shutdown if we got None as subscription request
        if sub_req is None:
            await close_everything(workers, exchanges)
            return

        # ..............................................................................
        sub_req_json = sub_req.to_json()

        # subscribe to topic, create worker & exchange if needed
        if action_str == "subscribe":
            try:
                await create_worker(sub_req, workers, exchanges)
            except Exception as e:
                logger.error("unable to create worker: %s", e)
            else:
                topics[sub_req_json] = topics[sub_req_json] + 1 \
                    if sub_req_json in topics else 1

        # unsubscribe from topic, remove worker & exchange if needed
        elif action_str == "unsubscribe":

            if sub_req_json not in topics:
                logger.warning(
                    "got unsubscribe for non-existent topic: %s", sub_req_json
                )
                return

            topics[sub_req_json] -= 1

            if topics[sub_req_json] == 0:
                logger.info("removing topic: %s", sub_req_json)
                del topics[sub_req_json]

                try:
                    await remove_worker(sub_req, workers, exchanges)
                except Exception as e:
                    logger.error("unable to remove worker: %s", e)

        # this should never happen, but just in case ...
        else:
            raise ValueError(f"invalid action: {action}")

    return stream_manager


# --------------------------------------------------------------------------------------
async def streamer(context: ContextT, config: ConfigT):
    ctx = context or zmq.asyncio.Context()

    async with Gond(config, ctx) as g:  # noqa: F841
        manager = get_stream_manager()

        # ..............................................................................
        while True:
            try:

                pass

            except asyncio.CancelledError:
                logger.info("Cancelled...")
                break

            except Exception as e:
                logger.exception(e)
                await asyncio.sleep(SLEEP_ON_ERROR)

        # tell the manager to pack it up ...
        manager(b"", None)

    # if counter > 0:
    #     duration = perf_counter() - start
    #     msg_per_sec = counter / duration
    #     sec_per_msg = duration / counter

    #     logger.info(
    #         "processed %s messages in %s seconds (%s msgs/s)",
    #         counter, duration, msg_per_sec
    #     )
    #     logger.info("one message every %s milliseconds", sec_per_msg / 1000)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    try:
        asyncio.run(streamer(None, None))
    except KeyboardInterrupt:
        print("Interrupted...")
