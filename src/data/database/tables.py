#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 202 17:43:33 2024

@author dhaneor
"""
import asyncio
import logging
from typing import Any
from .base import BaseTable, DatabaseManager

logger = logging.getLogger(f"main.{__name__}")


class OhlcvTable(BaseTable):
    def __init__(self, db_manager: DatabaseManager, name: str):
        super().__init__(db_manager)
        self.table_name = name
        logger.info(f"Creating OHLCV table: {self.table_name}")

    async def create(self):
        query = (
            """
            CREATE TABLE IF NOT EXISTS `%s` (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            openTime VARCHAR(15) NOT NULL UNIQUE,
            humanOpenTime VARCHAR(32) NOT NULL,
            open VARCHAR(32) NOT NULL,
            high VARCHAR(32) NOT NULL,
            low VARCHAR(32) NOT NULL,
            close VARCHAR(32) NOT NULL,
            volume VARCHAR(32) NOT NULL,
            closeTime BIGINT NOT NULL,
            quoteAssetVolume VARCHAR(32) NOT NULL,
            numberOfTrades INT,
            takerBuyBaseAssetVolume VARCHAR(32),
            takerBuyQuoteAssetVolume VARCHAR(32)
            ) """ % self.table_name
        )
        await self.db.execute(query)

    async def insert(table, data, retries=3):
        """
        Insert one or multiple rows into the table.

        Arguments:
        ----------
        data
            - A list of values for a single row (List[Any])
            - A list of rows, where each row is a list of values (List[List[Any]])
        """
        if not data:
            logger.warning("No data to insert.")
            return  # No data to insert

        for attempt in range(retries):
            try:
                await super().insert(data)
                logger.debug(
                    "INSERT OK [%s] for %s",
                    len(data) if isinstance(data[0], list) else 1,
                    table.table_name
                    )
                return
            except ValueError as ve:
                logger.error(
                    "[%s] INSERT FAIL for %s -> %s ",
                    attempt, table.table_name, str(ve)
                    )
                raise
            except Exception as e:
                logger.error(
                    "[%s] INSERT FAIL for %s -> %s ",
                    attempt, table.table_name, str(e)
                    )
                await asyncio.sleep(2)
                if attempt == retries - 1:  # Last retry
                    logger.error("🚨 All retries failed for insert.")
                    raise

    async def fetch_latest(self, limit: int = 1000):
        query = f"""
        SELECT * FROM {self.table_name}
        ORDER BY 'open time' DESC
        LIMIT :limit
        """
        return await self.db.fetch_all(query, {"limit": limit})

    async def fetch_by_range(self, start: int, end: int) -> list[list[Any]]:
        query = f"""
        SELECT * FROM {self.table_name}
        WHERE openTime BETWEEN :start AND :end
        ORDER BY 'open time' DESC
        """
        return await self.db.fetch_all(query, {"start": start, "end": end})
