#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the central data repository.

Hermes is the ancient god of wisdom and knowledge. He knows everything about:

    • OHLCV data for all cryptocurrencies (on all exchanges supported by CCXT)
    • Market (Symbol) information, like trading pairs, currencies, etc.

TODO:
    • Sentiment (future feature)
    • On-chain data (future feature)

Created on July 06 21:12:20 2023

@author dhaneor
"""
import logging
import os
from typing import Any
from dotenv import load_dotenv

from .exchange_factory import ExchangeFactory
from .ohlcv_repository import process_request
from .markets_repository import get_markets, get_symbols
from .data_models import Ohlcv, Markets, Market
from .database import DatabaseManager
from .util.timestamp_converter import timestamp_converter

logger = logging.getLogger(f"main.{__name__}")

# Load environment variables from .env file
load_dotenv()

# determine if we should use the database for OHLCV data,
# and initialize the database if necessary
USE_DB = os.getenv("USE_DB", "False").lower() == "true"
db = DatabaseManager() if USE_DB else None
logger.info(f"{'WILL NOT' if not USE_DB else 'WILL'} USE THE DATABASE")

# initialize the exchange factory
exchange_factory = ExchangeFactory()


class OhlcvRepository:
    """Manage and retrieve OHLCV data from various data sources."""
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
        return await process_request(req=request)


class MarketsRepository:

    async def all(self, exchange: str) -> Markets:
        req = {'exchange': exchange, 'data_type': 'markets'}
        return await get_markets(req)

    async def market(self, exchange: str, symbol: str) -> Market:
        return await self.all(exchange).get(symbol)

    async def all_symbols(self, exchange: str) -> list[str]:
        req = {'exchange': exchange, 'data_type': 'symbols'}
        return await get_symbols(req)


class Hermes:
    """
    Central data repository for cryptocurrency market information.

    This class provides a unified interface to access OHLCV
    (Open, High, Low, Close, Volume) data
    and market information for various cryptocurrencies across different exchanges.

    Attributes:
        ohlcv (OhlcvRepository): Repository for OHLCV data.
        markets (MarketsRepository): Repository for market information.

    The class is designed to be used as an asynchronous context manager, which handles
    database connections and cleanup of resources.

    Example:
        async with Hermes() as hermes:
            ohlcv_data = await hermes.get_ohlcv(
                exchange='binance', symbol='BTC/USDT', interval='1h'
                start=-1000, end='now UTC')
            markets_data = await hermes.get_markets('binance')

    Note:
        This class relies on environment variables and global settings:
        - USE_DB: Determines whether to use a database for data storage.
        - exchange_factory: A factory for creating exchange instances.
    """

    def __init__(self):
        """Initialize Hermes with OHLCV and Markets repositories."""
        self._ohlcv = OhlcvRepository()
        self._markets = MarketsRepository()

    async def __aenter__(self) -> 'Hermes':
        """
        Asynchronous enter method for context management.

        Establishes necessary connections when entering the context.

        Returns:
            Hermes: The Hermes instance.
        """
        if USE_DB:
            await db.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Asynchronous exit method for context management.

        Closes all exchange connections and disconnects from the database if used.
        Logs any errors that occurred during the context.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
            exc_val: The instance of the exception that caused the context to be exited.
            exc_tb: A traceback object encoding the stack trace.
        """
        await exchange_factory.close_all()

        if USE_DB:
            await db.disconnect()

        if exc_type is not None:
            logger.error(f"An error occurred: {exc_val}")

    async def ohlcv(self, **kwargs) -> Ohlcv:
        """
        Retrieve OHLCV data based on the provided parameters.

        Args:
            **kwargs: Keyword arguments specifying the OHLCV data request parameters.

        Returns:
            Ohlcv: An object containing the requested OHLCV data.
        """
        return await self._ohlcv.get_ohlcv(**kwargs)

    async def markets(self, exchange: str) -> Markets:
        """
        Retrieve all markets for a given exchange.

        Args:
            exchange (str): The name of the exchange.

        Returns:
            Markets: An object containing all markets for the specified exchange.
        """
        return await self._markets.all(exchange)

    async def market(self, exchange: str, symbol: str) -> Market:
        """
        Retrieve market information for a specific symbol on a given exchange.

        Args:
            exchange (str): The name of the exchange.
            symbol (str): The trading symbol to retrieve market information for.

        Returns:
            Market: An object containing market information for the specified symbol.
        """
        return await self._markets.market(exchange, symbol)

    async def symbols(self, exchange: str) -> list[str]:
        """
        Retrieve all trading symbols for a given exchange.

        Args:
            exchange (str): The name of the exchange.

        Returns:
            list[str]: A list of all trading symbols for the specified exchange.
        """
        return await self._markets.all_symbols(exchange)
