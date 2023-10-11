#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a wrapper for the websocket client for the Kucoin API.

This module uses the websocket client implementation from the
kucoin-python library/SDK.

These are the Kucoin limits for websocket connections:

Number of connections per user ID: ≤ 50
Connection Limit: 30 per minute
Message limit sent to the server: 100 per 10 seconds
Maximum number of batch subscriptions at a time: 100 topics
Subscription limit for each connection: 300 topics

Created on Sun Nov 28 13:12:20 2022

@author dhaneor
"""
import time
import asyncio
import logging
from typing import Union, List, Tuple, Callable, Optional

from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient

from .i_websockets import IWebsocketPublic, IWebsocketPrivate  # noqa: E402, F401
from .publishers import (  # noqa: E402, F401
    IPublisher,
    LogPublisher,
    ZeroMqPublisher,
    PrintPublisher,
)

TICKERS_ENDPOINT = "/market/ticker"
CANDLES_ENDPOINT = "/market/candles"
SNAPSHOT_ENDPOINT = "/market/snapshot"


# =============================================================================
class KucoinWebsocketPublic:
    """Provides a websocket connection for the Kucoin API.

    This is the depraceted version which is replaced by classes below
    that are now split into separate classes for tickers, OHLCV, etc.
    This makes it possible to have more topics and still stay within
    the limits of the Kucoin API (see module docstring).
    """

    publisher: IPublisher = PrintPublisher()

    def __init__(
        self,
        publisher: Optional[IPublisher] = None,
        callback: Optional[Callable] = None,
        id: Optional[str] = None,
    ):
        """Initialize a KucoinWebsocketPublic.

        Parameters
        ----------
        publisher : Optional[IPublisher], optional
            A class tht publishes the updates/messages, by default None
        callback : Optional[Callable], optional
            Alternative/additional function for publisher, by default None
        id : Optional[str], optional
            identifier for this instance, by default None
        """
        # set the callable to be used for publishing the results
        self.publisher = publisher or KucoinWebsocketPublic.publisher
        self.publish = callback or self.publisher.publish

        logger_name = f"main.{__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

        self.logger.info(
            f"kucoin public websocket initialized ..."
            f"with publisher {self.publisher}"
        )

        self.id = id

    async def run(self):
        try:
            await self.start_client()
        except asyncio.CancelledError:
            pass

    async def start_client(self):
        # is public
        # client = WsToken()
        # is private
        # client = WsToken(key='', secret='', passphrase='', is_sandbox=False, url='')
        # is sandbox
        # client = WsToken(is_sandbox=True)
        client = WsToken()

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            self.logger.error("unable to get running loop")
            loop = None

        try:
            self.ws_client = await KucoinWsClient.create(
                loop, client, self._handle_message, private=False
            )
            self.logger.info("kucoin public websocket client started ...")
        except asyncio.CancelledError:
            self.logger.info("websocket client stopped")

    async def watch_ticker(self, symbols: str | list[str] | None = None):
        if not self.ws_client:
            await self.start_client()

        symbols, symbols_str = await self._transform_symbols_parameter(symbols)

        if self.ws_client:
            try:
                await self.ws_client.subscribe(f"{TICKERS_ENDPOINT}:{symbols_str}")
                if symbols:
                    [
                        self.logger.info(f"subscribed to ticker stream for: {s} ")
                        for s in symbols
                    ]
            except Exception as e:
                self.logger.exception(e)
        else:
            raise Exception("watch ticker failed: missing client")

    async def unwatch_ticker(self, symbols: str | list[str] | None = None):
        symbols, symbols_str = await self._transform_symbols_parameter(symbols)

        if self.ws_client:
            try:
                await self.ws_client.unsubscribe(f"{TICKERS_ENDPOINT}:{symbols_str}")
                if symbols:
                    [
                        self.logger.info(f"unsubscribed from ticker stream for: {s} ")
                        for s in symbols
                    ]
            except ValueError:
                self.logger.error(
                    "someone tried to unsubscribe from a topic where we "
                    "had no subscription anyway"
                )
            except Exception as e:
                self.logger.exception(e)

    async def watch_candles(self, symbols: str | list[str], interval: str):
        if not self.ws_client:
            await self.start_client()

        symbols, _ = await self._transform_symbols_parameter(symbols)

        try:
            [
                await self.ws_client.subscribe(
                    f"{CANDLES_ENDPOINT}:{symbol}_{interval}"
                )
                for symbol in symbols
                if self.ws_client
            ]
            [
                self.logger.info(f"subscribed to kline stream for: {s}_{interval} ")
                for s in symbols
            ]
        except Exception as e:
            self.logger.exception(e)

    async def unwatch_candles(self, symbols: str | list[str], interval: str):
        if not self.ws_client:
            self.logger.error(
                "unwatch candles not possible as we have no subscriptions"
            )
            return

        symbols, _ = await self._transform_symbols_parameter(symbols)

        try:
            [
                await self.ws_client.unsubscribe(
                    f"{CANDLES_ENDPOINT}:{symbol}_{interval}"
                )
                for symbol in symbols
                if self.ws_client
            ]
            [
                self.logger.info(f"unsubscribed from kline stream for: {s} ")
                for s in symbols
            ]
        except ValueError:
            self.logger.error(
                "someone tried to unsubscribe from a topic where"
                " we had no subscription anyway"
            )
        except Exception as e:
            self.logger.exception(e)

    async def watch_snapshot(self, symbols: str | list[str] | None = None):
        if not self.ws_client:
            await self.start_client()

        symbols, _ = await self._transform_symbols_parameter(symbols)

        # subscribe based on market(s)
        if not symbols:
            self.logger.debug("subscribing to market snapshots ...")
            try:
                markets = ["USDS", "BTC", "ALTS", "KCS"]
                [
                    await self.ws_client.subscribe(f"{SNAPSHOT_ENDPOINT}:{m}")
                    for m in markets
                    if self.ws_client
                ]
                [
                    self.logger.info(f"subscribed to snapshot stream for market: {m}")
                    for m in markets
                ]
            except Exception as e:
                self.logger.exception(e)

        # subscribe based on specific symbols(s)
        else:
            self.logger.debug(f"subscribing to symbol snapshots ... {symbols}")
            try:
                [
                    await self.ws_client.subscribe(f"{SNAPSHOT_ENDPOINT}:{s}")
                    for s in symbols
                    if self.ws_client
                ]
                [
                    self.logger.info(f"subscribed to snapshot stream for: {s} ")
                    for s in symbols
                ]
            except Exception as e:
                self.logger.exception(e)

    async def unwatch_snapshot(self, symbols: str | list[str]):
        if not self.ws_client:
            self.logger.error(
                "unwatch candles not possible as we have no subscriptions"
            )
            return

        symbols, _ = await self._transform_symbols_parameter(symbols)

        try:
            [
                await self.ws_client.unsubscribe(f"{SNAPSHOT_ENDPOINT}:{symbol}")
                for symbol in symbols
                if self.ws_client
            ]
            [
                self.logger.info(f"unsubscribed from snapshot stream for: {s} ")
                for s in symbols
            ]
        except ValueError as e:
            self.logger.error(
                "someone tried to unsubscribe from a topic (%s) where"
                " we had no subscription anyway -> %s", symbols, e
            )
        except Exception as e:
            self.logger.exception(e)

    async def _handle_message(self, msg: dict) -> None:
        """Handles all messages the we get from the websocket client.

        A fairly long method. but this helps with performance. I
        measured the time for letting a specific method for every 'subject'
        handle the response and it was significantly slower, so I put
        everything in here. In the most demanding case (subscribing to all
        tickers or snapshots) we have thousands of messages per second,
        that's why we wanna be fast here.

        Parameters
        ----------
        msg
            the original websocket message
        """
        if msg["type"] == "message":
            subject = msg.get("subject")
            received_at = time.time()

            # ohlcv candle updates
            if subject and "candles" in subject:
                """
                .. code:: python
                # the original message format

                {
                    'data': {
                        'candles': [
                            '1669652580',
                            '16063.1',
                            '16058.2',
                            '16063.2',
                            '16056.7',
                            '2.90872302',
                            '46711.615964094'
                        ],
                        'symbol': 'BTC-USDT',
                        'time': 1669652610226195133
                    },
                    'subject': 'trade.candles.update',
                    'topic': '/market/candles:BTC-USDT_1min',
                    'type': 'message'
                }
                """
                await self.publish(
                    {
                        "id": self.id,
                        "exchange": "kucoin",
                        "subject": "candles",
                        "topic": msg["topic"].split(":")[-1],
                        "type": msg["subject"].split(".")[-1],
                        "symbol": msg["data"]["symbol"],
                        "interval": msg["topic"].split(":")[1].split("_")[1][:-2],
                        "data": {
                            "open time": msg["data"]["candles"][0],
                            "open": msg["data"]["candles"][1],
                            "high": msg["data"]["candles"][3],
                            "low": msg["data"]["candles"][4],
                            "close": msg["data"]["candles"][2],
                            "volume": msg["data"]["candles"][5],
                            "quote volume": msg["data"]["candles"][6],
                        },
                        "time": msg["data"]["time"] / 1_000_000_000,
                        "received_at": received_at,
                    }
                )
                return

            # ticker updates
            elif subject and "ticker" in subject:
                """
                .. code:: python
                # the original message format

                {
                    "type":"message",
                    "topic":"/market/ticker:BTC-USDT",
                    "subject":"trade.ticker",
                    "data":{
                        "sequence":"1545896668986",
                        "price":"0.08",
                        "size":"0.011",
                        "bestAsk":"0.08",
                        "bestAskSize":"0.18",
                        "bestBid":"0.049",
                        "bestBidSize":"0.036"
                    }
                }
                """
                try:
                    await self.publish(
                        {
                            "exchange": "kucoin",
                            "subject": "ticker",
                            "topic": msg["topic"].split(":")[-1],
                            "data": msg["data"],
                            "received_at": received_at,
                            "type": "message"
                        }
                    )
                    return
                except Exception:
                    return

            # market snapshot updates
            elif subject and "snapshot" in subject:
                """
                .. code:: python
                # the original message format

                {
                    'type': 'message',
                    'topic': '/market/snapshot:UNI-USDT',
                    'subject': 'trade.snapshot',
                    'data': {
                        'sequence': '169723469',
                        'data': {
                            'averagePrice': 6.04811259,
                            'baseCurrency': 'UNI',
                            'board': 1,
                            'buy': 5.9512,
                            'changePrice': 0.05,
                            'changeRate': 0.0084,
                            'close': 5.9536,
                            'datetime': 1670487368010,
                            'high': 6.0521,
                            'lastTradedPrice': 5.9536,
                            'low': 5.8732,
                            'makerCoefficient': 1.0,
                            'makerFeeRate': 0.001,
                            'marginTrade': True,
                            'mark': 0,
                            'market': 'DeFi',
                            'markets': ['DeFi', 'USDS'],
                            'open': 5.9036,
                            'quoteCurrency': 'USDT',
                            'sell': 5.9536,
                            'sort': 100,
                            'symbol': 'UNI-USDT',
                            'symbolCode': 'UNI-USDT',
                            'takerCoefficient': 1.0,
                            'takerFeeRate': 0.001,
                            'trading': True,
                            'vol': 41390.5055,
                            'volValue': 247542.9903228
                            }
                        }
                    }
                """
                try:
                    msg["topic"] = msg["topic"].split(":")[1]
                except Exception:
                    pass

                try:
                    msg["exchange"] = "kucoin"
                    msg["subject"] = "snapshot"
                    msg["received_at"] = received_at
                    await self.publish(msg)
                except Exception as e:
                    self.logger.error(f"unable to handle message: {msg} {e}")
                return

            # probably an update from watching all tickers where subject
            # refers to the symbol name
            elif subject:
                try:
                    await self.publish(
                        {
                            "exchange": "kucoin",
                            "subject": "ticker",
                            "topic": msg["topic"].split(":")[-1],
                            "symbol": msg["subject"],
                            "data": msg["data"],
                            "type": "ticker",
                            "received_at": received_at,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"unable to handle message: {msg} {e}")
                return

        # .....................................................................
        # something went wrong and we got an error message
        elif msg["type"] == "error":
            """
            when trying to subscribe to a non-existent topic, we get
            an error message like this one:
            {
                'id': '1670421943842',
                'type': 'error',
                'code': 404,
                'data': 'topic /market/candles:abc-USDT_1min is not found'
            }
            """
            try:
                wrong_topic = msg["data"].split(":")[1].split(" ")[0]
            except Exception:
                wrong_topic = False

            if wrong_topic:
                error_msg = {
                    "type": "error",
                    "error": "subscription failed",
                    "code": msg["code"],
                    "topic": wrong_topic,
                }
            else:
                error_msg = {
                    "type": "error",
                    "error": "unknown",
                    "code": msg["code"],
                    "topic": msg["topic"],
                }

            self.logger.error(error_msg)
            await self.publish(error_msg)
            return

        # .....................................................................
        # all other messages that are not covered by the cases above,
        # we should never get here ...
        else:
            self.logger.error(f"unable to handle message: {msg}")

    async def _transform_symbols_parameter(
        self, symbols: Union[List[str], str, None]
    ) -> Tuple[List[str], str]:
        if not symbols:
            return [], "all"

        if isinstance(symbols, list):
            return symbols, ",".join(symbols)
        elif isinstance(symbols, str):
            return [symbols], symbols
        else:
            raise ValueError(
                f"<symbols> parameter must be str or List[str]"
                f" but was {type(symbols)}"
            )


class KucoinWebsocketPrivate(IWebsocketPrivate):
    ws_client = None
    credentials = {}
    logger_name = f"main.{__name__}"
    publisher = LogPublisher(logger_name)

    def __init__(
        self,
        credentials: dict,
        callback: Callable,
        publisher: Union[IPublisher, None] = None,
    ):
        # set the callable to be used for publishing the results
        self.publisher = publisher or KucoinWebsocketPublic.publisher
        self.publish = callback or self.publisher.publish

        self.credentials = credentials
        self.logger = logging.getLogger(self.logger_name)

    async def run(self):
        await self.start_client()
        await self.watch_account()

    async def start_client(self):
        # is public
        # client = WsToken()
        # is private
        client = WsToken(
            key=self.credentials["api_key"],
            secret=self.credentials["api_secret"],
            passphrase=self.credentials["api_passphrase"],
            is_sandbox=False,
            url="",
        )
        # is sandbox
        # client = WsToken(is_sandbox=True)
        # client = WsToken()

        self.ws_client = await KucoinWsClient.create(
            None, client, self._handle_message, private=True
        )

    async def watch_account(self):
        await self.watch_balance()
        await self.watch_orders()
        await self.watch_debt_ratio()

    async def unwatch_account(self):
        await self.unwatch_balance()
        await self.unwatch_orders()

    async def watch_balance(self):
        if not self.ws_client:
            await self.start_client()

        if self.ws_client:
            try:
                await self.ws_client.subscribe("/account/balance")
                self.logger.debug("subscribed to balance stream")
            except Exception as e:
                self.logger.exception(e)

    async def unwatch_balance(self):
        if self.ws_client:
            try:
                await self.ws_client.unsubscribe("/account/balance")
                self.logger.debug("unsubscribed from balance stream")
            except Exception as e:
                self.logger.exception(e)

    async def watch_orders(self):
        if not self.ws_client:
            await self.start_client()

        if self.ws_client:
            try:
                await self.ws_client.subscribe("/spotMarket/tradeOrders")
                self.logger.info("subscribed to orders stream")
                await self.ws_client.subscribe("/spotMarket/advancedOrders")
                self.logger.info("subscribed to advanced orders stream")
            except Exception as e:
                self.logger.exception(e)

    async def unwatch_orders(self):
        if self.ws_client:
            try:
                await self.ws_client.unsubscribe("/spotMarket/tradeOrders")
                self.logger.info("unsubscribed from order stream")
                await self.ws_client.subscribe("/spotMarket/advancedOrders")
                self.logger.info("unsubscribed from advanced orders stream")
            except Exception as e:
                self.logger.exception(e)

    async def watch_debt_ratio(self):
        if not self.ws_client:
            await self.start_client()

        if self.ws_client:
            try:
                await self.ws_client.subscribe("/margin/position")
                self.logger.debug("subscribed to debt ratio stream")
            except Exception as e:
                self.logger.exception(e)

    async def unwatch_debt_ratio(self):
        if self.ws_client:
            try:
                await self.ws_client.unsubscribe("/margin/position")
                self.logger.debug("unsubscribed from debt ratio stream")
            except Exception as e:
                self.logger.exception(e)

    async def _handle_message(self, msg):
        if msg["type"] == "message":
            subject = msg.get("subject")

            # account balance updates
            if subject and subject == "account.balance":
                await self.publish(msg)
                return

            # order updates
            elif subject and subject == "orderChange":
                await self.publish(msg)
                return

            # advanced (stop) order updates
            elif subject and subject == "stopOrder":
                await self.publish(msg)
                return

            # debt ratio updates
            elif subject and subject == "debt.ratio":
                await self.publish(msg)
                return
        else:
            self.logger.error(f"unable to handle message: {msg}")
