#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:23:58 2022

@author: dhaneor
"""
import logging
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Union, Tuple, List, Dict, Callable, Type


from broker.models.requests import PositionChangeRequest
from broker.models.exchange_order import ExchangeOrder
from ..position_handlers.actions import Action

from ..ganesh import Ganesh
from .balance import Balance, balance_factory
from .symbol import Symbol


# ==============================================================================
class Position(ABC):
    """Abstract Base class for all position types."""

    TYPE = ""
    ENTRY_SIDE = ""
    EXIT_SIDE = ""

    def __init__(self, broker: Ganesh):

        self.broker: Ganesh = broker

        self.asset: str
        self.balance: Balance
        self.quote_asset: Union[str, None]
        self.symbol: Union[Symbol, None]
        self._last_price: float = 0

        self.asset_precision: int

        self.entry_orders: Union[list, None] = None
        self.exit_orders: Union[list, None] = None
        self.active_stop_orders: Dict[str, dict] = {}
        self.active_limit_orders: Dict[str, dict] = {}

        self.request: Union[PositionChangeRequest, None] = None
        self.pending_action: Union[Action, None] = None
        self.handler: Union[object, None] = None

        self.logger: logging.Logger  # will be set by the PositionFactory

    def __repr__(self):
        try:
            value = self.get_value(quote_asset=self.quote_asset)
        except:
            value = 0
        net_balance = round(self.balance.net)
        if self.symbol:
            net_balance = round(net_balance, self.symbol.lot_size_step_precision)
        else:
            net_balance = round(net_balance, 8)

        res = f"{self.asset} \t{self.TYPE} position: "
        res += f"{net_balance: <12} "
        res += f"(value: {value} {self.quote_asset}) "
        # res += f'\t({(", ").join(self.symbols_for_asset)})'

        if self.entry_orders:
            res += f"\nentry orders:\n"
            for order in self.entry_orders:
                res += f"\t{order}\n"
        if self.exit_orders:
            res += f"\nexit orders:\n"
            for order in self.exit_orders:
                res += f"\t{order}\n"

        return res

    # --------------------------------------------------------------------------
    @property
    def symbols_for_asset(self):
        return tuple(
            sym["symbol"]
            for sym in self.broker.symbols
            if sym.get("baseAsset") == self.asset
        )

    @property
    def non_zero(self):
        return self.balance.net != 0

    @property
    def is_dust(self):
        if self.asset == self.quote_asset:
            return False

        return abs(self.balance.net) < self.symbol.lot_size_min  # type: ignore

    @property
    def last_price(self):
        return self._last_price if self._last_price else None

    @last_price.setter
    def last_price(self, value: Union[float, str]):
        try:
            self._last_price = float(value)
            self.logger.debug(
                f"updated price for {self.asset}: {value} {self.quote_asset}"
            )
        except:
            self.logger.error(
                f"cannot set last price from {value} {type(value)}" f"for {self.asset}"
            )

    @property
    def value(self):
        return self.get_value(self.quote_asset)

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_value(self, quote_asset: Union[str, None]) -> float:
        pass

    @abstractmethod
    def get_history(self):
        pass

    # --------------------------------------------------------------------------
    def accept_consultant(self):
        return self

    def add_request(self, request: Union[PositionChangeRequest, None]):
        self.request = request

    def handle_stop_order_update(self, msg: dict):
        """
        example message:
        {
            'type': 'message',
            'topic': '/spotMarket/advancedOrders',
            'userId': '5fd10f949910b40006395f9e',
            'channelType': 'private',
            'subject': 'stopOrder',
            'data': {
                'createdAt': 1670183280902,
                'orderId': 'vs93qoscv5o3qkrg000hhb61',
                'orderType': 'stop',
                'side': 'sell',
                'size': '9.8414',
                'stop': 'loss',
                'stopPrice': '0.39132',
                'symbol': 'XRP-USDT',
                'tradeType': 'MARGIN_TRADE',
                'ts': 1670183340836885829,
                'type': 'cancel'
            }
        }
        """
        try:
            if self.symbol and msg["symbol"] == self.symbol.name:
                order_id = msg["orderId"]
                if msg["type"] == "open":
                    self.active_stop_orders[order_id] = msg
                    self.logger.debug(f"added order {order_id} to active stop orders")
                    return
                if msg["type"] == "cancel":
                    del self.active_stop_orders[order_id]
                    self.logger.debug(
                        f"removed order {order_id} from active stop orders"
                    )
                    return
                if "trigger" in msg["type"].lower():
                    del self.active_stop_orders[order_id]
                    self.logger.debug(
                        f"removed order {order_id} from active stop orders"
                    )
                    return

            else:
                raise ValueError(f"stop order update for wrong symbol")
        except Exception as e:
            self.logger.error(f"failed to handle stop order update: {msg} -> {e}")
            raise ValueError(e)


# ------------------------------------------------------------------------------
class LongPosition(Position):

    TYPE = "LONG"
    ENTRY_SIDE = "BUY"
    EXIT_SIDE = "SELL"

    def __init__(self, broker: Ganesh):
        super().__init__(broker)
        # self.position_detective = LongPositionDetective(self)
        # self.position_detective.do_your_thing()

    # --------------------------------------------------------------------------
    @property
    def _active_take_profit_orders(self):
        return self.active_limit_orders

    # --------------------------------------------------------------------------
    def get_value(self, quote_asset: Union[str, None]):

        if quote_asset == self.balance.asset:
            return self.balance.net

        if not quote_asset:
            return self.balance.net

        if not self.symbol:
            return 0

        if self.last_price:
            value = self.balance.net * self.last_price
        else:
            value = 0

        if not quote_asset or quote_asset == self.symbol.quote_asset:
            return round(value, self.symbol.quote_precision)

        else:
            quote_symbol_name = f"{quote_asset}-{self.quote_asset}"
            value /= self.broker.get_last_price(quote_symbol_name)
            symbol = self.broker.get_symbol(quote_symbol_name)
            return round(value, symbol.quote_precision)  # type: ignore

    def get_history(self):
        pass


class ShortPosition(Position):

    TYPE = "SHORT"
    ENTRY_SIDE = "SELL"
    EXIT_SIDE = "BUY"

    def __init__(self, broker: Ganesh):
        super().__init__(broker)

    def get_value(self, quote_asset: Union[str, None]):
        if quote_asset == self.balance.asset:
            return self.balance.net

        if not self.symbol:
            return 0

        if self.last_price:
            value = self.balance.net * self.last_price
        else:
            value = 0

        if not quote_asset or quote_asset == self.symbol.quote_asset:
            return round(value, self.symbol.quote_precision)  # type: ignore

        else:
            quote_symbol_name = f"{quote_asset}-{self.quote_asset}"
            value /= self.broker.get_last_price(quote_symbol_name)
            symbol = self.broker.get_symbol(quote_symbol_name)
            return round(value, symbol.quote_precision)  # type: ignore

    def get_history(self):
        pass


class ZeroPosition(Position):

    TYPE = "NO"

    def __init__(self, broker: Ganesh):
        super().__init__(broker)

        self.TYPE = ""

    def get_value(self, quote_asset: Union[str, None] = None):
        return 0

    def get_history(self):
        pass


# ==============================================================================
class PositionDetective(ABC):
    """Position Detective helps to deduct useful information for positions.

    This class is intented to be used exclusively with the Position
    classes defined above. The idea is to have it as component of the
    Position classes, but it could also be used by the Account class
    to operate externally on the positions.

    This class provides methods to find entries and exits for instance.
    This then helps to calculate the PNL for the position - without
    relying on any other data source than what we get from the exchange
    API (balance, orders, ...).

    NOTE:   This approach was chosen because the whole broker subsystem
            is built in a way that it can be deployed as a single
            container on a cloud instance.

            The goal is to be state-less and therefore to be independent
            from any kind of database which would increase deployment/
            operating costs, possibly increase execution time and
            reduce flexibility regarding deployment.

            The underlying reason is, that it should be easy to have
            more than one instance (one for each supported exchange)
            or move to another cloud provider and/or region, for
            instance to achieve a minimal distance to the exchange
            servers for reduced latency.
            Speed (and therefore latency) is important because it
            should be able to operate on multiple symbols as fast as we
            can after the 'Close', possibly even for multiple different
            users.
    """

    def __init__(self, position: Position):
        self.position = position
        self.broker = position.broker

    # --------------------------------------------------------------------------
    @property
    def orders(self):
        return [
            o for o in self.broker.orders if o.symbol in self.position.symbols_for_asset
        ]

    @property
    def active_stop_orders(self):
        return [
            o
            for o in self.broker.orders
            if all(
                arg
                for arg in (
                    o.symbol in self.position.symbols_for_asset,
                    o.status == "NEW",
                    "STOP" in o.type,
                )
            )
        ]

    @property
    def active_limit_orders(self):
        return [
            o
            for o in self.broker.orders
            if all(
                arg
                for arg in (
                    o.symbol in self.position.symbols_for_asset,
                    o.status == "NEW",
                    "LIMIT" in o.type,
                )
            )
        ]

    # --------------------------------------------------------------------------
    def do_your_thing(self):
        self._add_entry_orders()
        self._add_exit_orders()

    # --------------------------------------------------------------------------
    def _add_entry_orders(self):
        self.position.entry_orders = [
            o
            for o in self.orders
            if all(
                arg
                for arg in (o.side == self.position.ENTRY_SIDE, o.status == "FILLED")
            )
        ]

    def _add_exit_orders(self):
        self.position.exit_orders = [
            o
            for o in self.orders
            if all(
                arg for arg in (o.side == self.position.EXIT_SIDE, o.status == "FILLED")
            )
        ]


# ------------------------------------------------------------------------------
class LongPositionDetective(PositionDetective):

    def __init__(self, position: Position):
        super().__init__(position)


class ShortPositionDetective(PositionDetective):

    def __init__(self, position: Position):
        super().__init__(position)


# ==============================================================================
class PositionFactory:
    """Factory Class to build Position objects."""

    def __init__(self, broker: Ganesh):
        self.broker = broker
        self.quote_asset: str = ""
        self.all_active_stop_orders: Union[Tuple[ExchangeOrder], None]
        self.stop_order_download_done: bool = False
        self.logger = logging.getLogger("main.PositionFactory")

    def build_positions(self, account: List[dict], quote_asset: str) -> Tuple[Position]:
        """Builds all positions for all valid assets.

        :param quote_asset: the quote asset to use for the account
        :type quote_asset: str
        :return: all positions
        :rtype: Tuple[Position]
        """
        self.quote_asset = quote_asset
        valid_assets = self.broker.valid_assets
        valid_quote_assets = self.broker.valid_quote_assets

        try:
            return tuple(
                self.build_position_from_balance(balance_factory(bal))
                for bal in account
                if any(
                    (
                        bal["currency"] in valid_assets,
                        bal["currency"] in valid_quote_assets,
                    )
                )
            )
        except Exception as e:
            self.logger.exception(e)
            return tuple()

    def build_position_from_balance(self, balance: Balance) -> Position:
        """Builds a Position object from a balance dictionary.

        :param balance: a Balance object for a single asset
        :type balance: Balance
        :return: List of complete Position objects
        :rtype: Position

        Expects a dictionary representing the balance for a single
        asset in the following format:
        """
        position = self._initialize_position_from_balance(balance)
        position.balance = balance
        position.asset = balance.asset
        position.quote_asset = self.quote_asset
        position.logger = logging.getLogger(f"main.Position.{position.asset}")

        if position.asset == position.quote_asset:
            position.symbol = None
            position.last_price = 1
        else:
            try:
                position.symbol = self.broker.get_symbol(
                    base_asset=balance.asset, quote_asset=self.quote_asset
                )
            except Exception as e:
                position.symbol = None
                print(
                    f"failed to build position from balance for "
                    f"{balance.asset}: {e}"
                )

        try:
            if position.symbol:
                position.asset_precision = position.symbol.lot_size_step_precision
            else:
                currency = next(
                    filter(
                        lambda x: x["currency"] == position.asset,
                        self.broker.currencies,
                    )
                )
                position.asset_precision = currency["precision"]
        except Exception as e:
            self.logger.exception(e)

        try:
            if position.symbol:
                return self._add_active_stop_orders_to_position(position)
        except Exception as e:
            self.logger.error(f"failed to add stop orders to position {position} - {e}")

        return position

    def _initialize_position_from_balance(self, balance: Balance) -> Position:
        """Initializes the correct Positon class for LONG/SHORT positions.

        :param balance: _description_
        :type balance: Balance
        :return: _description_
        :rtype: Position
        """
        if balance.net > 0:
            return LongPosition(broker=self.broker)
        elif balance.net < 0:
            return ShortPosition(broker=self.broker)
        else:
            return ZeroPosition(broker=self.broker)

    def _add_active_stop_orders_to_position(self, position: Position):
        if not self.stop_order_download_done:
            self.logger.debug("getting all active stop orders")
            self.all_active_stop_orders = self.broker.get_active_stop_orders()
            self.stop_order_download_done = True
            if self.all_active_stop_orders:
                self.logger.debug(f"got {len(self.all_active_stop_orders)} orders")

        if not self.all_active_stop_orders:
            self.logger.debug("no stop orders, no adding to position")
            return position

        for o in self.all_active_stop_orders:
            if position.symbol and position.symbol.name == o.symbol:
                position.active_stop_orders[o.order_id] = {
                    "createdAt": int(o.time / 1000),
                    "orderId": o.order_id,
                    "orderType": "stop",
                    "side": o.side,
                    "size": o.orig_qty - o.executed_base_qty,
                    "stop": "entry" if o.type == "STOP_ENTRY_MARKET" else "loss",
                    "stopPrice": o.stop_price,
                    "symbol": o.symbol,
                    "tradeType": "MARGIN_TRADE",
                    "ts": int(o.time),
                    "type": "open",
                }
                self.logger.debug(f"added stop order {o} to {position}")

        return position

        """
        the attributes of ExchangeOrder:
        {
            'client_order_id': 'fv2VPLSKO3EImxInscsISf',
            'executed_base_qty': 0.0,
            'executed_quote_qty': 0.0,
            'fill_price': 0.0,
            'fills': [],
            'order_id': 'vs93qosd1ob6lmhp000tubbn',
            'orig_qty': 9.4459,
            'orig_quote_qty': 0.0,
            'price': 0.39122,
            'side': 'BUY',
            'status': 'NEW',
            'stop_price': 0.39122,
            'symbol': 'XRP-USDT',
            'time': 1670188566032.0,
            'time_human': '2022-12-04 21:16:06',
            'time_in_force': 'GTC',
            'type': 'STOP_ENTRY_MARKET'
        }
        """
