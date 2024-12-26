#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the central data repository.

Hermes is the ancient god of wisdom and knowledge. He knows everything about:

• OHLCV data for all cryptocurrencies (on all exchanges supported by CCXT)
• Symbol information (like trading pairs, currencies, etc.)

• Sentiment (future feature)
• On-chain data (future feature)

Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import os
from functools import partial
from typing import Any
from dotenv import load_dotenv

from .exchange_factory import exchange_factory_fn
from .ohlcv_repository import process_request
from .markets_repository import get_markets
from .data_models import Ohlcv, Markets, Market
from .import database
from .util import timestamp_converter

logger = logging.getLogger(f"main.{__name__}")

# Load environment variables from .env file
load_dotenv()

# determine if we should use the database for OHLCV data,
# and initialize the database if necessary
USE_DB = os.getenv("USE_DB", "False").lower() == "true"
db = database.DatabaseManager() if USE_DB else None
logger.info(f"{'WILL NOT' if not USE_DB else 'WILL'} USE THE DATABASE")

# initialize the exchange factory
exchange_factory = exchange_factory_fn()
# prepare to make sure that all modules, that we are working with here,
# use the same instance of the exchange factory. this is necessary for
# a clean shutdown process.
process_request_fn = partial(process_request, exchange_factory=exchange_factory)
get_markets_fn = partial(get_markets, exchange_factory=exchange_factory)
database.process_request = process_request_fn


class OhlcvRepository:
    """
    Manage and retrieve OHLCV data from various data sources.
    """
    def __init__(self) -> None:
        ...

    @timestamp_converter(unit='milliseconds')
    async def get_ohlcv(self, **kwargs) -> Ohlcv:
        req = kwargs
        if req.get('exchange') == 'binance' and USE_DB:
            return await self._get_ohlcv_from_database(req)
        else:
            return await self._get_ohlcv_from_exchange(req)

    async def _get_ohlcv_from_database(self, req: dict[str, Any]) -> Ohlcv:
        table = await db.get_table(req['exchange'], req['symbol'], req['interval'])

        if not await table.exists():
            logger.info(
                "creating OHLCV table for %s...",
                f"{req['exchange']} {req['symbol']} {req['interval']}"
                )
            await table.create()

        return await table.fetch_by_range(req['start'], req['end'])

    async def _get_ohlcv_from_exchange(self, request: dict[str, Any]) -> Ohlcv:
        """Fetch OHLCV data for a given exchange and symbol.

        Parameters:
        exchange (str): The name of the exchange.
        symbol (str): The symbol for which OHLCV data is requested.
        interval (str): The interval for OHLCV data.
        start (int): The start timestamp for the OHLCV data.
        end (int): The end timestamp for the OHLCV data.
        as_dataframe (bool): Whether to return the data as a pandas DataFrame.

        Returns:
        Response: An OHLCV object containing the OHLCV data.
        """
        return await process_request_fn(req=request)


class MarketsRepository:

    async def all(self, exchange: str) -> Markets:
        req = {'exchange': exchange, 'data_type': 'markets'}
        return await get_markets_fn(req)

    async def market(self, exchange: str, symbol: str) -> Market:
        return self.all(exchange).get(symbol)


class Hermes:

    def __init__(self):
        self.ohlcv = OhlcvRepository()
        self.markets = MarketsRepository()

    async def __aenter__(self) -> 'Hermes':
        if USE_DB:
            await db.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            logger.error(f"An error occurred: {exc_val}")

        await exchange_factory.close_all()
        if USE_DB:
            await db.disconnect()

