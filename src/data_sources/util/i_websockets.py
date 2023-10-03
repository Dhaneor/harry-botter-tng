#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the interfaces for public and private websocket classes.


Created on Wed Dec 07 12:44:23 2022

@author_ dhaneor
"""
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

from .publishers import IPublisher


# ======================================================================================
class IWebsocketPublic(ABC):
    """Interface / absstract base class for websocket
    handlers for public (not requiring API keys) endpoints.
    """

    publish: Callable
    publisher: IPublisher

    @abstractmethod
    def __init__(
        self,
        publisher: IPublisher | None = None,
        callback: Callable | None = None,
        id: str | None = None,
    ):
        """Initializes websocket handler for public endpoints.

        Parameters
        ----------

        publisher: IPublisher | None
            class/object that handles the actual publishing of events
            received from the websocket stream. This can be configured
            dynamically to suit different needs. mplementations of
            this interface should set a default publisher for the case
            that both parameters are None. See publisher.py for some
            implentations. If not set, a callback function or method
            that handles received data should be provided.

        callback: Callable
            callback that processes ws data, defaults to None
        """
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start_client(self):
        pass

    @abstractmethod
    async def watch_ticker(self, symbols: Union[str, List[str], None] = None):
        pass

    @abstractmethod
    async def unwatch_ticker(self, symbols: Union[str, List[str], None] = None):
        pass

    @abstractmethod
    async def watch_candles(self, symbols: Union[str, List[str]], interval: str):
        pass

    @abstractmethod
    async def unwatch_candles(self, symbols: Union[str, List[str]], interval: str):
        pass

    @abstractmethod
    async def watch_snapshot(self, symbols: Union[str, List[str], None] = None):
        pass

    @abstractmethod
    async def unwatch_snapshot(self, symbols: Union[str, List[str]]):
        pass

    @abstractmethod
    async def _handle_message(self, msg: dict) -> None:
        pass


class IWebsocketPrivate(ABC):
    """Interface / absstract base class for websocket
    handlers for private (requiring API keys) endpoints.
    """

    publisher: IPublisher

    @abstractmethod
    def __init__(self, publisher: Union[IPublisher, None] = None):
        """Initializes websocket handler for private endpoints.

        :param publisher: class/object that handles the actual
        publishing of events received from the websocket stream.
        This can be configured dnymaically to suit different needs.
        Implementations of this interface should set a default
        publisher. See publisher.py for some implentations.
        :type publisher: Union[IPublisher, None], optional
        """
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start_client(self):
        pass

    @abstractmethod
    async def watch_account(self):
        pass

    @abstractmethod
    async def unwatch_account(self):
        pass

    @abstractmethod
    async def watch_balance(self):
        pass

    @abstractmethod
    async def unwatch_balance(self):
        pass

    @abstractmethod
    async def watch_orders(self):
        pass

    @abstractmethod
    async def unwatch_orders(self):
        pass

    @abstractmethod
    async def watch_debt_ratio(self):
        pass

    @abstractmethod
    async def unwatch_debt_ratio(self):
        pass

    @abstractmethod
    async def _handle_message(self, msg: dict) -> None:
        pass
