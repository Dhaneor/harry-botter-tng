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

from .rawi.exchange_factory import exchange_factory_fn
from .rawi import ohlcv_repository
from .rawi.util.timestamp_converter import timestamp_converter

logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.DEBUG)


class Hermes:

    def __init__(self):
        self.exchange_factory = exchange_factory_fn()

    async def __aenter__(self) -> 'Hermes':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.exchange_factory(None)

    @timestamp_converter(unit='milliseconds')
    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start: int | str,
        end: int | str,
    ) -> ohlcv_repository.Response:
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
        request = {
            'exchange': exchange,
            'symbol': symbol,
            'interval': interval,
            'start': start,
            'end': end,
        }

        response = await ohlcv_repository.process_request(
            request, self.exchange_factory
            )
        return response
