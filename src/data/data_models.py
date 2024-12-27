#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a specialized OHLCV repository for the OHLCV containers.

classes
    Response

    This class standardizes the response, does type checks for the
    values from the request, and includes a method for sending the
    response after fetching the OHLCV data (including some error
    flags for the client).

Created on Tue Dec 24 12:23:23 2024

@author_ dhaneor
"""
import datetime
import json
import logging
import numpy as np
import yaml

from dataclasses import dataclass
from typing import Any
from util import seconds_to, unix_to_utc

logger = logging.getLogger("main.data_models")

interval_in_ms = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000
}


# ====================================================================================
@dataclass
class Ohlcv:
    """Ohlcv data class.

    This class standardizes the response, does type checks for the values from
    the request, and includes a method for sending the response over the ZeroMQ
    socket (if provided) after fetching the OHLCV data from the database or the
    exchange API (including some error flags for the client).
    """

    exchange: str = None
    symbol: str = None
    interval: str = None
    start: int = None
    end: int = None
    socket: object = None
    id: str = None
    data: list[list[Any]] | None = None
    _execution_time: float | None = None
    _bad_request_error: bool = False
    _authentication_error: bool = False
    _exchange_error: bool = False
    _fetch_ohlcv_not_available: bool = False
    _symbol_error: bool = False
    _interval_error: bool = False
    _network_error: bool = False

    def __repr__(self):
        data_str, error_str = "", ""

        if self.data is not None and len(self.data) > 0:
            data_str = f", data[{len(self.data)}]: {self.data[1]}...{self.data[-1]}"

        if not self.success:
            error_str = f", bad request: {self.bad_request}, errors: {self.errors}"

        start = unix_to_utc(self.data[0][0]) if self.data else unix_to_utc(self.start)
        end = unix_to_utc(self.data[-1][0]) if self.data else unix_to_utc(self.end)

        return (
            f"Ohlcv(exchange={self.exchange}, symbol={self.symbol}, "
            f"interval={self.interval}, start={start}, end={end}"
            f"{error_str}{data_str})"
        )

    def __post_init__(self):
        self.exchange_error = True if not isinstance(self.exchange, str) else False
        self.symbol_error = True if not isinstance(self.symbol, str) else False
        self.interval_error = True if not isinstance(self.interval, str) else False

        # assume end is now, if not provided
        if self.end is None:
            print("No end time provided, assuming now")
            self.end = datetime.datetime.now(tz=datetime.UTC).timestamp() * 1000

        # assume 1000 intervals by default, if not provided
        if self.start is None:
            self.start = self.end - interval_in_ms[self.interval] * 1000

    @property
    def interval_in_ms(self):
        return interval_in_ms.get(self.interval, 0)

    @property
    def execution_time(self):
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float):
        self._execution_time = value

        logger.info(
            "Fetched %s elements for %s %s on %s in %s: %s",
            len(self.data) if self.data else 0,
            self.symbol,
            self.interval,
            self.exchange,
            seconds_to(value),
            "OK" if self.data else "FAIL",
        )

    @property
    def errors(self):
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("_") and "error" in attr and getattr(self, attr)
        }

    @property
    def authentication_error(self):
        return self._authentication_error

    @authentication_error.setter
    def authentication_error(self, value: bool):
        self._authentication_error = value

    @property
    def exchange_error(self):
        return self._exchange_error

    @exchange_error.setter
    def exchange_error(self, value: bool):
        self._exchange_error = value

    @property
    def fetch_ohlcv_not_available(self):
        return self._fetch_ohlcv_not_available

    @fetch_ohlcv_not_available.setter
    def fetch_ohlcv_not_available(self, value: bool):
        self._fetch_ohlcv_not_available = value

    @property
    def symbol_error(self):
        return self._symbol_error

    @symbol_error.setter
    def symbol_error(self, value: bool):
        self._symbol_error = value

    @property
    def interval_error(self):
        return self._interval_error

    @interval_error.setter
    def interval_error(self, value: bool):
        self._interval_error = value

    @property
    def network_error(self):
        return self._network_error

    @network_error.setter
    def network_error(self, value: bool) -> None:
        self._network_error = True

    @property
    def bad_request(self) -> bool:
        return (
            True
            if any(
                (
                    self._bad_request_error,
                    self._authentication_error,
                    self._exchange_error,
                    self._symbol_error,
                    self._interval_error,
                )
            )
            else False
        )

    @bad_request.setter
    def bad_request(self, value: bool) -> None:
        self._bad_request_error = value

    @property
    def success(self) -> bool:
        return False if self.bad_request else True

    # ------ Functions for sending the response and reconstructing it from JSON ------
    @classmethod
    def from_json(cls, json_string: str) -> "Ohlcv":
        """
        Reconstruct a Response object from a JSON string.

        Parameters:
        -----------
        json_string : str
            A JSON string containing the serialized Response data.

        Returns:
        --------
        Response
            A new Response object reconstructed from the JSON data.
        """
        json_data = json.loads(json_string)

        response = cls(
            exchange=json_data.get("exchange"),
            symbol=json_data.get("symbol"),
            interval=json_data.get("interval"),
            start=json_data.get("start"),
            end=json_data.get("end"),
            socket=None,  # Socket can't be serialized, so we set it to None
            id=None,
        )

        # Restore other attributes
        response.data = json_data.get("data", [])

        # Reconstruct errors
        errors = json_data.get("errors", {})
        errors = errors if isinstance(errors, dict) else {}
        for error_type, error_message in errors.items():
            if error_message:  # Only set non-False error messages
                setattr(response, error_type, error_message)

        return response

    def to_json(self):
        response = {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "success": self.success,
            "data": self.data,
            "bad_request": self.bad_request,
            "errors": self.errors or None,
        }

        # prevent crashes due to unserializable values
        for k, v in response.items():
            try:
                json.dumps(v)
            except Exception as e:
                logger.warning(f"Could not serialize {k} value: {v} ({e})")
                response[k] = None
                response["errors"][k] = str(e)
                response["success"] = False
        return response

    async def send(self):
        if not self.socket:
            logger.error("Cannot send response: socket is not set")
            return

        # Send the response back through the socket
        await self.socket.send_multipart(
            [self.id, b"", json.dumps(self.to_json()).encode("utf-8")]
        )

    # -------- Functions for converting the OHLCV data to the desired format ---------
    def to_dict(self) -> dict[str, np.ndarray] | None:
        """Convert OHLCV data to a dictionary.

        Returns
        -------
        Optional[dict[str, Any]]
            A dictionary containing the OHLCV data, or None if we have no data
        """
        if not self.data:
            return None

        data_array = np.array(self.data).T

        return {
            "open time": data_array[0],
            "open": data_array[1],
            "high": data_array[2],
            "low": data_array[3],
            "close": data_array[4],
            "volume": data_array[5],
        }

    def _timestamp_to_datetime(self, timestamp: int) -> datetime:
        """Convert a Unix timestamp to a datetime object."""
        return datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)


# ====================================================================================
#                               Symbols informmation                                 #
# ====================================================================================
class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        return self.to_yaml()

    def __str__(self):
        return self.to_yaml()

    def to_yaml(self):
        """Convert the DotDict to a YAML string, including nested values."""
        return yaml.dump(
            self.to_dict(), default_flow_style=False, sort_keys=False, indent=4
            )

    def to_dict(self):
        """Recursively convert DotDict (and nested DotDicts) to standard dictionaries"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):  # Check if the value is a DotDict
                result[key] = value.to_dict()  # Recursively convert nested DotDict
            else:
                result[key] = value if key != 'info' else None
        return result


class Market(DotDict):
    ...


@dataclass
class Symbols:
    """Symbols data class.

    This class standardizes the response, does type checks for the values from
    the request, and includes a method for sending the response over the ZeroMQ
    socket (if provided) after fetching the symbols data from the database or the
    exchange API (including some error flags for the client).
    """

    exchange: str = None
    socket: object = None
    id: str = None
    data: list[str] | None = None
    _execution_time: float | None = None
    _bad_request_error: bool = False
    _authentication_error: bool = False
    _exchange_error: bool = False
    _network_error: bool = False

    def __repr__(self):
        if self.data is not None and len(self.data) > 0:
            data_str = f", data[{len(self.data)}]: {self.data[0]}...{self.data[-1]}"
        else:
            data_str = f", errors: {self.errors}"

        success_str = (
            f", bad request: {self.bad_request}"
            if self.bad_request
            else f", success: {self.success}"
        )

        return f"Symbols(exchange={self.exchange}{success_str}{data_str})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, item: str) -> bool:
        return item in self.data

    @property
    def execution_time(self):
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float):
        self._execution_time = value

        logger.info(
            "Fetched %s symbols for %s in %s: %s",
            len(self.data) if self.data else 0,
            self.exchange,
            seconds_to(value),
            "OK" if self.data else "FAIL",
        )

    @property
    def errors(self):
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("_") and "error" in attr and getattr(self, attr)
        }

    @property
    def authentication_error(self):
        return self._authentication_error

    @authentication_error.setter
    def authentication_error(self, value: bool):
        self._authentication_error = value

    @property
    def exchange_error(self):
        return self._exchange_error

    @exchange_error.setter
    def exchange_error(self, value: bool):
        self._exchange_error = value

    @property
    def network_error(self):
        return self._network_error

    @network_error.setter
    def network_error(self, value: bool) -> None:
        self._network_error = True

    @property
    def bad_request(self) -> bool:
        return (
            True
            if any(
                (
                    self._bad_request_error,
                    self._authentication_error,
                    self._exchange_error,
                )
            )
            else False
        )

    @bad_request.setter
    def bad_request(self, value: bool) -> None:
        self._bad_request_error = value

    @property
    def success(self) -> bool:
        return False if self.bad_request else True

    def __post_init__(self):
        # Basic checks on instantiation
        self.exchange_error = True if not isinstance(self.exchange, str) else False

    # ------ Functions for sending the response and reconstructing it from JSON ------
    @classmethod
    def from_json(cls, json_string: str) -> "Symbols":
        """
        Reconstruct a Symbols object from a JSON string.

        Parameters:
        -----------
        json_string : str
            A JSON string containing the serialized Symbols data.

        Returns:
        --------
        Symbols
            A new Symbols object reconstructed from the JSON data.
        """
        json_data = json.loads(json_string)

        symbols = cls(
            exchange=json_data.get("exchange"),
            socket=None,  # Socket can't be serialized, so we set it to None
            id=None,
        )

        # Restore other attributes
        symbols.data = json_data.get("data", [])

        # Reconstruct errors
        errors = json_data.get("errors", {})
        errors = errors if isinstance(errors, dict) else {}
        for error_type, error_message in errors.items():
            if error_message:  # Only set non-False error messages
                setattr(symbols, error_type, error_message)

        return symbols

    def to_json(self):
        response = {
            "exchange": self.exchange,
            "success": self.success,
            "data": self.data,
            "bad_request": self.bad_request,
            "errors": self.errors or None,
        }

        # prevent crashes due to unserializable values
        for k, v in response.items():
            try:
                json.dumps(v)
            except Exception as e:
                logger.warning(f"Could not serialize {k} value: {v} ({e})")
                response[k] = None
                response["errors"][k] = str(e)
                response["success"] = False
        return response

    async def send(self):
        if not self.socket:
            logger.error("Cannot send response: socket is not set")
            return

        # Send the response back through the socket
        await self.socket.send_multipart(
            [self.id, b"", json.dumps(self.to_json()).encode("utf-8")]
        )

    # -------- Functions for converting the symbols data to the desired format ---------
    def to_dict(self) -> dict[list[str]]:
        """Convert symbols data to a dictionary.

        Returns
        -------
        dict[list[str]]
            A dictionary containing the symbols data, or None if we have no data
        """
        return {"symbols": self.data if self.data else []}


@dataclass
class Markets(Symbols):
    """Markets data class.

    This class standardizes the response, does type checks for the values from
    the request, and includes a method for sending the response over the ZeroMQ
    socket (if provided) after fetching the markets data from the database or the
    exchange API (including some error flags for the client).
    """

    def __repr__(self):
        if self.data is not None and len(self.data) > 0:
            keys = list(sorted(self.data.keys()))
            data_str = f", data[{len(self.data)}]: {keys[0]}...{keys[-1]}"
        else:
            data_str = f", errors: {self.errors}"

        success_str = (
            f", bad request: {self.bad_request}"
            if self.bad_request
            else f", success: {self.success}"
        )

        return f"Markets(exchange={self.exchange}{success_str}{data_str})"

    def __contains__(self, item: str) -> bool:
        return item in [symbol["symbol"] for symbol in self.data]

    def __getitem__(self, item: str) -> dict[str, Any]:
        return self.data.get(item, {})

    @property
    def execution_time(self):
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float):
        self._execution_time = value

        logger.info(
            "Fetched %s markets for %s in %s: %s",
            len(self.data) if self.data else 0,
            self.exchange,
            seconds_to(value),
            "OK" if self.data else "FAIL",
        )

    def get(self, symbol: str, default: Any) -> dict[str, Any] | Any:
        if symbol := self.data.get(symbol):
            return symbol
        else:
            return default

    # -------- Functions for converting the markets data to the desired format ---------
    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert markets data to a dictionary.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            A dictionary containing the markets data, or None if we have no data
        """
        return {"markets": self.data if self.data else {}}


if __name__ == "__main__":
    sym_info = {
        'active': True,
        'base': 'BTC',
        'baseId': 'BTC',
        'contract': False,
        'contractSize': None,
        'created': None,
        'expiry': None,
        'expiryDatetime': None,
        'future': False,
        'id': 'BTC-USDT',
        'index': None,
        'info': {
            'baseCurrency': 'BTC',
            'baseIncrement': '0.00000001',
            'baseMaxSize': '10000000000',
            'baseMinSize': '0.00001',
            'callauctionFirstStageStartTime': None,
            'callauctionIsEnabled': False,
            'callauctionPriceCeiling': None,
            'callauctionPriceFloor': None,
            'callauctionSecondStageStartTime': None,
            'callauctionThirdStageStartTime': None,
            'enableTrading': True,
            'feeCategory': 1,
            'feeCurrency': 'USDT',
            'isMarginEnabled': True,
            'makerFeeCoefficient': '1.00',
            'market': 'USDS',
            'minFunds': '0.1',
            'name': 'BTC-USDT',
            'priceIncrement': '0.1',
            'priceLimitRate': '0.1',
            'quoteCurrency': 'USDT',
            'quoteIncrement': '0.000001',
            'quoteMaxSize': '99999999',
            'quoteMinSize': '0.1',
            'st': False,
            'symbol': 'BTC-USDT',
            'takerFeeCoefficient': '1.00',
            'tradingStartTime': None},
        'inverse': None,
        'limits': {
            'amount': {'max': 10000000000.0, 'min': 1e-05},
            'cost': {'max': 99999999.0, 'min': 0.1},
            'leverage': {'max': None, 'min': None},
            'price': {'max': None, 'min': None}},
        'linear': None,
        'lowercaseId': None,
        'maker': 0.001,
        'margin': True,
        'marginMode': {'cross': False, 'isolated': False},
        'marginModes': {'cross': None, 'isolated': None},
        'option': False,
        'optionType': None,
        'percentage': True,
        'precision': {
            'amount': 1e-08,
            'base': None,
            'cost': None,
            'price': 0.1,
            'quote': None
        },
        'quote': 'USDT',
        'quoteId': 'USDT',
        'settle': None,
        'settleId': None,
        'spot': True,
        'strike': None,
        'subType': None,
        'swap': False,
        'symbol': 'BTC/USDT',
        'taker': 0.001,
        'tierBased': True,
        'tiers': {
            'maker': [
                [0.0, 0.001],
                [50.0, 0.0009],
                [200.0, 0.0007],
                [500.0, 0.0005],
                [1000.0, 0.0003],
                [2000.0, 0.0],
                [4000.0, 0.0],
                [8000.0, 0.0],
                [15000.0, -5e-05],
                [25000.0, -5e-05],
                [40000.0, -5e-05],
                [60000.0, -5e-05],
                [80000.0, -5e-05]],
            'taker': [
                [0.0, 0.001],
                [50.0, 0.001],
                [200.0, 0.0009],
                [500.0, 0.0008],
                [1000.0, 0.0007],
                [2000.0, 0.0007],
                [4000.0, 0.0006],
                [8000.0, 0.0005],
                [15000.0, 0.00045],
                [25000.0, 0.0004],
                [40000.0, 0.00035],
                [60000.0, 0.0003],
                [80000.0, 0.00025]
                ]
            },
        'type': 'spot'}

    symbol = Market(sym_info)

    print(symbol)
    print(symbol.precision.amount)
    print(f"{'precision' in symbol}")

    o = Ohlcv(exchange='binance', symbol='Btc/USDT', interval='1d')
    print(o)
