#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 202 15:50:33 2024

@author dhaneor
"""
import logging
import os
from abc import abstractmethod
from databases import Database
from dotenv import load_dotenv
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(f"main.{__name__}")

# Load environment variables from .env file
load_dotenv()

logger.debug(f"DB_USER: {os.getenv("DB_USER")}")
logger.debug(f"DB_PASSWORD: {os.getenv("DB_PASS")}")
logger.debug(f"DB_HOST: {os.getenv("DB_HOST")}")
logger.debug(f"DB_PORT: {os.getenv("DB_PORT")}")
logger.debug(f"DB_NAME: {os.getenv("DB_NAME", "akasha")}")


class DatabaseManager:
    def __init__(self):
        db_user = os.getenv("DB_USER")
        db_password = str(os.getenv("DB_PASS"))
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "akasha")

        if not db_user or not db_password:
            raise EnvironmentError("DB_USER and DB_PASSWORD environment variables must be set.")

        # Encode the username
        encoded_user = quote_plus(db_user)

        self._db_url = f"mysql+aiomysql://{encoded_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logger.debug(f"Database URL: {self._db_url}")
        self._database = Database(self._db_url)

    async def connect(self):
        try:
            await self._database.connect()
        except Exception as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False
        logger.debug("✅ Connected to the MySQL database.")
        return True

    async def disconnect(self):
        await self._database.disconnect()
        logger.debug("🛑 Disconnected from the MySQL database.")

    @property
    def db(self):
        return self._database


class BaseTable:
    table_name: str = ""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager.db
        self.columns: list[str] = []

    async def exists(self) -> bool:
        query = f"SHOW TABLES LIKE '{self.table_name}'"
        result = await self.db.fetch_one(query)
        return bool(result)

    @abstractmethod
    async def create(self):
        ...

    async def drop(self) -> None:
        query = f"DROP TABLE IF EXISTS {self.table_name}"
        await self.db.execute(query)

    async def insert(
        self,
        data: list[Any] | list[list[Any]],
    ) -> None:
        await self._fetch_columns()  # Ensure columns are fetched and cached

        columns_formatted = ', '.join(self.columns)
        placeholders = ', '.join(['%s'] * len(self.columns))
        query = (
            f"INSERT IGNORE INTO {self.table_name} "
            f"({columns_formatted}) VALUES ({placeholders})"
        )

        # Check if data is a single row or multiple rows
        if isinstance(data[0], list):
            logger.debug(f"BATCH INSERT for {self.table_name} with {len(data)} entries")
            # Batch insert
            if len(data[0]) != len(self.columns) - 1:
                logger.debug(data[0])
                raise ValueError(
                    f"Length of data {len(data[0])} and number of "
                    f"columns {len(self.columns) - 1} does not match."
                    )

            await self.db.execute_many(query, data)
        else:
            # Single row insert
            if len(data) != len(self.columns) - 1:
                raise ValueError(
                    f"Length of data {len(data)} and number of "
                    f"columns {len(self.columns) - 1} does not match."
                    )

            await self.db.execute(query, data)

    async def fetch_all(self) -> list[list[Any]]:
        query = f"SELECT * FROM {self.table_name}"
        return await self.db.fetch_all(query)

    async def fetch_by(self, **conditions) -> list[dict[str, Any]]:
        where_clause = " AND ".join([f"{key} = :{key}" for key in conditions.keys()])
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        return await self.db.fetch_all(query, conditions)

    # ................................................................................
    async def _fetch_columns(self):
        """
        Fetch column names from the database schema and cache them.
        """
        if self.columns:
            return self.columns  # Return cached columns

        query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{self.table_name}'
        """
        result = await self.db.fetch_all(query)
        self.columns = [row['COLUMN_NAME'] for row in result]
        if not self.columns:
            raise ValueError(f"No columns found for table '{self.table_name}'.")
