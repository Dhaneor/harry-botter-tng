#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 202 18:00:33 2024

@author dhaneor
"""

import asyncio
import logging
import os
import sys

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------
from src.data.database.tables import OhlcvTable  # noqa: E402
from src.data.database.base import DatabaseManager  # noqa: E402

# Set up logging
logger = logging.getLogger("main")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Dummy data for testing
dummy_candles = [
    [
        1609459200000,
        "10:00",
        100.0,
        105.0,
        98.0,
        102.0,
        1000.0,
        1609459259999,
        102000.0,
        100,
        500.0,
        51000.0,
    ],
    [
        1609545600000,
        "11:00",
        102.0,
        107.0,
        101.0,
        106.0,
        1200.0,
        1609545659999,
        127200.0,
        120,
        600.0,
        63600.0,
    ],
    [
        1609632000000,
        "12:00",
        106.0,
        110.0,
        104.0,
        108.0,
        1100.0,
        1609632059999,
        118800.0,
        110,
        550.0,
        59400.0,
    ],
]


async def test_ohlcv_table():
    # Initialize DatabaseManager and OhlcvTable
    db_manager = DatabaseManager()
    connected = await db_manager.connect()
    if not connected:
        logger.error("Failed to connect to the database.")
        return
    ohlcv_table = OhlcvTable(db_manager, "BINANCE_NOCOIN_1d")

    # Test create method
    logger.info("Testing create method...")
    await ohlcv_table.create()
    logger.info("Table created successfully.")

    # Test insert method (single and batch)
    logger.info("Testing insert method...")
    try:
        await ohlcv_table.insert(dummy_candles)
    except Exception as e:
        logger.error(f"Failed to insert candles: {e}", exc_info=True)
        await db_manager.disconnect()
        return
    logger.info("Candles inserted successfully.")

    # Test fetch_latest method
    logger.info("Testing fetch_latest method...")
    latest_candles = await ohlcv_table.fetch_latest(limit=5)
    logger.info(f"Latest candles: {latest_candles}")

    # Test fetch_by_range method
    logger.info("Testing fetch_by_range method...")
    range_candles = await ohlcv_table.fetch_by_range(
        start=1609459200000, end=1609632000000
    )
    logger.info(f"Candles in range: {range_candles}")

    # Test base class methods
    logger.info("Testing base class methods...")

    # Test _fetch_columns method
    await ohlcv_table._fetch_columns()
    logger.info(f"Fetched columns: {ohlcv_table.columns}")

    # Test execute method
    logger.info("Testing execute method...")
    query = f"SELECT COUNT(*) FROM {ohlcv_table.table_name}"
    result = await ohlcv_table.db.execute(query)
    logger.info(f"Number of rows in table: {result}")

    # Test fetch method
    logger.info("Testing fetch method...")
    fetched_data = await ohlcv_table.fetch_latest()
    logger.info(f"Fetched data: {fetched_data}")

    logger.info("Testing drop method...")
    await ohlcv_table.drop()
    logger.info("Table dropped successfully.")

    # Close the database connection
    await db_manager.disconnect()

    logger.info("All tests completed successfully.")


# Run the test function
if __name__ == "__main__":
    asyncio.run(test_ohlcv_table())
