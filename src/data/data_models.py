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
import json
import logging
import numpy as np

from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("main.ohlcv_respsonse")


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
        if self.data is not None and len(self.data) > 0:
            data_str = f", data[{len(self.data)}]: {self.data[1]}...{self.data[-1]}"
        else:
            data_str = f", errors: {self.errors}"

        return (
            f"Response(exchange={self.exchange}, symbol={self.symbol}, "
            f"interval={self.interval}, start={self.start}, end={self.end}, "
            f"success={self.success}, bad_request={self.bad_request}{data_str})"
        )

    @property
    def execution_time(self):
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float):
        self._execution_time = value

        logger.info(
            "Fetched %s elements for %s %s in %s ms: %s",
            len(self.data) if self.data else 0,
            self.symbol,
            self.interval,
            int(value * 1000),
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

    def __post_init__(self):
        # Basic checks on instantiation
        self.exchange_error = True if not isinstance(self.exchange, str) else False
        self.symbol_error = True if not isinstance(self.symbol, str) else False
        self.interval_error = True if not isinstance(self.interval, str) else False

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
