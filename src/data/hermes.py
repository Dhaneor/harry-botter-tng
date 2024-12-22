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
import pandas as pd

from .rawi.exchange_factory import exchange_factory_fn
from .rawi.ohlcv_repository import Response
from .rawi import ohlcv_repository
from .util.timestamp_converter import timestamp_converter

logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.DEBUG)

exchange_facctory = exchange_factory_fn()


class Hermes:

    def __init__(self):
        ...

    @timestamp_converter(unit='milliseconds')
    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start: int,
        end: int,
        as_dataframe: bool = True,
    ) -> Response:
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
        # Fetch the OHLCV data from the appropriate exchange
        return Response(exchange=exchange, symbol=symbol, interval=interval)
