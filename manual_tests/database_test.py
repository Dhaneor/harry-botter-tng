#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:00:33 2024

@author dhaneor
"""

import asyncio

from data.database import DatabaseManager, OhlcvTable
from data import ohlcv_repository as repo
from data.exchange_factory import exchange_factory_fn
from util.logger_setup import get_logger

logger = get_logger(level="INFO")

# Dummy data for testing
dummy_candles = [
    [
        1609459200000,
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


# ====================================================================================
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
        logger.error(f"Failed to insert candles: {e}", exc_info=False)
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
    logger.info(f"Candles in range: {len(range_candles)}")
    for candle in range_candles:
        logger.info(f"{candle[0]}: {candle[1:]}")

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


async def test_ohlcv_table_with_real_data() -> None:
    # initialize exchange_factory
    exchange_factory = exchange_factory_fn()

    # download OHLCV data from Binance
    request = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "interval": "4h",
        "start": "six years ago UTC",
        "end": "one day ago UTC",
    }

    try:
        response = await repo.process_request(request, exchange_factory)
    except Exception as e:
        logger.error(f"Failed to download OHLCV data: {e}")
        await exchange_factory(None)  # close the exchange connection
        return

    logger.debug(f"Downloaded OHLCV data: {response}")
    logger.debug(f"Number of candles: {len(response.data)}")

    data_types = [type(elem) for elem in response.data[0]]
    logger.debug(f"Data types: {data_types}")

    # close the exchange connection
    await exchange_factory(None)

    # Initialize DatabaseManager and OhlcvTable
    db_manager = DatabaseManager()
    connected = await db_manager.connect()
    if not connected:
        logger.error("Failed to connect to the database.")
        return
    ohlcv_table = OhlcvTable(db_manager, "binance", response.symbol, response.interval)

    # Test create method
    logger.info("Testing create method...")
    await ohlcv_table.create()
    logger.info("Table created successfully.")

    # Test insert method (single and batch)
    logger.info("Testing insert method...")
    try:
        await ohlcv_table.insert(response.data)
    except Exception as e:
        logger.error(f"Failed to insert candles: {e}", exc_info=False)
        await ohlcv_table.drop()
        await db_manager.disconnect()
        return
    logger.info("Candles inserted successfully.")

    # test get_row_count method
    logger.info("Testing get_row_count method...")
    row_count = await ohlcv_table.get_row_count()
    logger.info(f"Number of rows in table: {row_count}")

    # Test fetch_latest method
    latest = await ohlcv_table.fetch_latest(limit=5)
    for candle in latest:
        logger.info(f"{candle[0]}: {candle[1:]}")

    # test needs_update method
    logger.info("Testing needs_update method...")
    needs_update = await ohlcv_table.needs_update()
    logger.info(f"Table needs to be updated: {needs_update}")

    logger.info("All tests completed successfully.")

    await ohlcv_table.drop()
    await db_manager.disconnect()


async def test_ohlcv_table_update() -> None:
    # initialize exchange_factory & set it for the repository as well,
    # to make sure that everyrhing is closed properly at the end
    exchange_factory = exchange_factory_fn()
    repo.EXCHANGE_FACTORY = exchange_factory

    # Initialize DatabaseManager and OhlcvTable
    db_manager = DatabaseManager()
    connected = await db_manager.connect()
    if not connected:
        logger.error("Failed to connect to the database.")
        return

    symbol = "BTC/USDT"
    interval = "4h"

    # download OHLCV data from Binance
    request = {
        "exchange": "binance",
        "symbol": symbol,
        "interval": interval,
        "start": "six years ago UTC",
        "end": "one day ago UTC",
    }

    try:
        ohlcv_table = await db_manager.get_table("binance", symbol, interval)
        await ohlcv_table.create()

        response = await repo.process_request(request, exchange_factory)
        await ohlcv_table.insert(response.data)

        row_count = await ohlcv_table.get_row_count()
        logger.info(f"Number of rows in table: {row_count}")

        # Test fetch_latest method
        latest = await ohlcv_table.fetch_latest(limit=5)
        for candle in latest.data:
            logger.info(f"{candle[0]}: {", ".join([str(e) for e in candle[1:6]])} ...")

    except Exception as e:
        logger.exception(e)
    else:
        logger.info("All tests completed successfully.")
    finally:
        await ohlcv_table.drop()
        await db_manager.disconnect()

        # close the exchange connection
        await exchange_factory(None)


# Run the test function
if __name__ == "__main__":
    asyncio.run(test_ohlcv_table_update())
