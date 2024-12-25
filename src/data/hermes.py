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
from typing import Any
from dotenv import load_dotenv

from .exchange_factory import exchange_factory_fn
from .ohlcv_repository import process_request
from .data_models import Ohlcv as OHlcvData
from .database import DatabaseManager
from .util.timestamp_converter import timestamp_converter

logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.DEBUG)

exchange_factory = exchange_factory_fn()

# Load environment variables from .env file
load_dotenv()

# determine if we should use the database for OHLCV data
USE_DB = os.getenv("USE_DB", "False").lower() == "true"

logger.info(
    f"{'WILL NOT' if not USE_DB else 'WILL'} USE THE DATABASE for OHLCV data."
    )

db = DatabaseManager() if USE_DB else None


class OhlcvRepository:
    """
    Manage and retrieve OHLCV data from various data sources.
    """
    def __init__(self):
        ...

    @timestamp_converter(unit='milliseconds')
    async def get_ohlcv(self, req: dict[str, Any]) -> OHlcvData:
        if req.get('exchange') == 'binance' and USE_DB:
            return await self._get_ohlcv_from_database(req)
        else:
            return await self._get_ohlcv_from_exchange(req)

    async def _get_ohlcv_from_database(req: dict[str, Any]) -> OHlcvData:
        table = await db.get_table(req['exchange'], req['symbol'], req['interval'])
        return await table.fetch_by_range(req['start'], req['end'])

    async def _get_ohlcv_from_exchange(request: dict[str, Any]) -> OHlcvData:
        """Fetch OHLCV data for a given exchange and symbol.

        Parameters:
        exchange (str): The name of the exchange.
        symbol (str): The symbol for which OHLCV data is requested.
        interval (str): The interval for OHLCV data.
        start (int): The start timestamp for the OHLCV data.
        end (int): The end timestamp for the OHLCV data.
        as_dataframe (bool): Whether to return the data as a pandas DataFrame.

        Returns:
        Response: A Response object containing the OHLCV data.
        """
        return await process_request(request, exchange_factory)


class SymbolsRepository:
    ...


class Hermes:

    def __init__(self):
        self.ohlcv = OhlcvRepository()

    async def __aenter__(self) -> 'Hermes':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await exchange_factory(None)
        await db.disconnect()
