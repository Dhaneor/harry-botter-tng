#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:54:23 2022

@author: dhaneor
"""
import os
import time
import sys
import os
import asyncio
import concurrent
import logging

from abc import ABC, abstractmethod
from threading import Thread, Lock
from typing import Union, Tuple

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from ..abstract_broker import (AbstractPrivateBroker,
                               AbstractPublicBroker,
                               AbstractPrivateSpotBroker,
                               AbstractPrivateMarginBroker
)
from .event_bus import (EventBus, Event, OrderCreatedEvent,
                        OrderFilledEvent, OrderCancelledEvent,
                        LoanCreatedEvent, LoanRepaidEvent)
from broker.models.exchange_order import build_exchange_order, ExchangeOrder
from util.timeops import unix_to_utc

logger = logging.getLogger('main.repositories')
logger.setLevel(logging.INFO)

event_bus = EventBus()

public_client = AbstractPublicBroker
private_client = Union[AbstractPrivateMarginBroker, AbstractPrivateSpotBroker]

# ==============================================================================

class UpdateTimer(Thread):
    """Timer that calls a (callback) method after a given period of time.

    The run() method must be called once externally to start the timer.
    """

    def __init__(self, callback, ttl:int):
        """_summary_

        :param callback: callback function to notify when timer is triggered
        :type callback: function
        :param ttl: time-to-live = call callback every ttl seconds
        :type ttl: int
        """
        Thread.__init__(self, daemon=True, group=None)
        self._callback = callback
        self._ttl = ttl
        self._last_triggered = time.time()

    def run(self):
        while True:
            now = time.time()
            if now - self._last_triggered > self._ttl:
                self._callback()
                self.reset()
            time.sleep(self._ttl/10)

    def reset(self):
        self._last_triggered = time.time()


# ==============================================================================
class BaseRepository(Thread, ABC):

    def __init__(self, client, ttl:int=3600):

        Thread.__init__(self, daemon=True)

        self.client: Union[private_client, public_client] = client

        self.data: tuple
        self.ttl: int = ttl

        self._must_update: bool = True
        self.updating: bool = False

        self.update_timer = UpdateTimer(
            callback=self.update_before_next_call, ttl=self.ttl
            )
        self.update_timer.start()

        logger_name = 'main.repositories.' + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

    @property
    def must_update(self) -> bool:
        return True if self._must_update else False

    @must_update.setter
    def must_update(self, value: bool):
        if isinstance(value, bool):
            self._must_update = value
        else:
            raise TypeError('Please provide a boolean value for <must_update>')

    # --------------------------------------------------------------------------
    @abstractmethod
    def update(self):
        pass

    def post_update(self, data):
        self.data = data
        self.reset()

    def update_before_next_call(self):
        with Lock():
            self.updating = True
            self.update()
            self.updating = False

    def handle_event(self, event: Union[Event, None]=None):
        if isinstance(event, Event):
            self.must_update = True
            self.logger.debug(f'received Event {event}')
        else:
            self.logger.warning('expected Event but received: {event}')

    def reset(self):
        self.update_timer.reset()
        self.must_update = False

    # --------------------------------------------------------------------------
    def _register_with_event_bus(self, event_types:list):
            if event_types and event_types is not None:
                event_bus.register_subscriber(self, event_types)

    def _values_to_numeric(self, item:dict) -> dict:

        for k,v in item.items():
            try:
                item[k] = float(v)
            except:
                pass

        return item


# ==============================================================================
#                           Static Repositories                                #
# ==============================================================================
class SymbolRepository(BaseRepository):

    EVENT_TYPES = []

    def __init__(self, client:public_client, ttl:int=1440):

        BaseRepository.__init__(self, client, ttl)
        self.client: public_client = client

    @property
    def symbols(self) -> Tuple[dict]:
        if self.must_update:
            self.update()

        return tuple(self.data)

    def update(self):
        res = self.client.get_symbols()

        if res.get('success'):
            symbols = res.get('message')
        else:
            symbols = ()
            error = res.get('error', 'unkknown')
            self.logger.error(f'update failed: {error}')

        self.post_update(symbols)


class MarginConfigurationRepository(BaseRepository):

    EVENT_TYPES = []

    def __init__(self, client: object=None, ttl:int=1440):

        BaseRepository.__init__(self, client, ttl)

    @property
    def margin_configuration(self):
        if self.must_update:
            self.update()

        return self.data

    def update(self):
        res = self.client.get_margin_config()

        if res.get('success'):
            margin_configuration = res.get('message')
        else:
            margin_configuration = []
            error = '[_update_margin_configuration]: ' + res.get('error')
            self.logger.error(error)

        self.post_update(margin_configuration)

# ==============================================================================
#                           Dynamic Repositories                               #
# ==============================================================================
class TickerRepository(BaseRepository):

    EVENT_TYPES = []

    def __init__(self, client:public_client, ttl:int=30):

        BaseRepository.__init__(self, client, ttl)
        self.data: Tuple[dict]

    @property
    def tickers(self) -> Tuple[dict]:
        while self.updating:
            time.sleep(0.01)

        while self.must_update:
            self.update()

        return self.data

    def update(self):
        res = self.client.get_all_tickers()
        if res.get('success'):
            tickers = res.get('message')
            self.logger.debug('TickerRepository updated')
        else:
            tickers = ()
            error = '[_update_tickers]: ' + res.get('error')
            self.logger.error(error)

        self.post_update(tickers)


class RiskLimitRepository(BaseRepository):

    EVENT_TYPES = []

    def __init__(self, client: public_client, ttl:int=1440):

        BaseRepository.__init__(self, client, ttl)
        self.client: public_client
        self.data: tuple

    @property
    def risk_limits(self) -> Tuple[dict]:
        while self.updating:
            time.sleep(0.05)

        while self.must_update:
            self.update()

        return self.data

    def update(self):
        res = self.client.get_margin_risk_limit()

        if res.get('success'):
            risk_limits = res.get('message')
            risk_limits = tuple(
                self._values_to_numeric(item) for item in risk_limits
            )
            self.logger.debug('RiskLimitRepository updated')
        else:
            risk_limits = ()
            error = '[update] for risk limits: ' + res.get('error')
            self.logger.error(error)

        self.post_update(risk_limits)


class AccountRepository(BaseRepository):
    """Class to get the complete ACCOUNT on the exchange from.

    The ACCOUNT is the list of all balances in the currently used
    market.

    NOTE:   In general the market could be 'SPOT', 'CROSS MARGIN',
            'ISOLATED MARGIN' or 'FUTURES', depending on the exchange.
            For now only Kucoin and the 'CROSS MARGIN' market are
            fully implemented and tested as this is what I want to
            use. Feel free to implement other options! :)
    """

    EVENT_TYPES = [
        OrderFilledEvent, OrderCancelledEvent, OrderCreatedEvent,
        LoanCreatedEvent, LoanRepaidEvent
    ]

    def __init__(self, client: private_client, ttl:int=30):

        BaseRepository.__init__(self, client, ttl)
        self._register_with_event_bus(AccountRepository.EVENT_TYPES)
        self.client: private_client

    @property
    def account(self):

        self.logger.debug('got request for account data')

        while self.updating:
            time.sleep(0.05)

        while self.must_update:
            self.update()

        return self.data

    @property
    def debt_ratio(self):
        while self.must_update:
            self.logger.debug('I need an update')
            self.update()

        return self._debt_ratio

    def update(self):
        res = self.client.get_account()

        if res.get('success'):
            message = res.get('message')
            self._debt_ratio = message.get('debtRatio')
            account = message.get('accounts')
            account = [self._values_to_numeric(item) for item in account]
            account = [self._add_net_balance(balance) for balance in account]

            self.logger.debug('AccountRepository updated')
        else:
            account = []
            error = '[AccountRepository.update()]: ' + res.get('error')
            self.logger.error(error)

        self.post_update(account)

    def _add_net_balance(self, balance):
        balance['net'] = balance['total'] - balance['borrowed']
        return balance


class BorrowDetailsRepository(BaseRepository):

    EVENT_TYPES = [
        OrderCreatedEvent, OrderFilledEvent, LoanCreatedEvent, LoanRepaidEvent
    ]

    def __init__(self, client:object=None, ttl:int=30):
        BaseRepository.__init__(self, client, ttl)
        self._register_with_event_bus(BorrowDetailsRepository.EVENT_TYPES)

        self._debt_ratio: float

    @property
    def borrow_details(self):
        while self.updating:
            time.sleep(0.05)

        while self.must_update:
            self.update()

        return self.data

    @property
    def debt_ratio(self) -> float:
        if self.must_update:
            self.update()

        return self._debt_ratio

    @debt_ratio.setter
    def debt_ratio(self, value: float):
        self._debt_ratio = value

    # --------------------------------------------------------------------------
    def update(self):
        res = self.client.get_borrow_details_for_all()

        if res.get('success'):
            message = res.get('message')
            borrow_details = message.get('accounts', [])
            borrow_details = [
                self._values_to_numeric(item) for item in borrow_details
                ]

            self.debt_ratio = message.get('debtRatio')
        else:
            borrow_details = []
            error = '[BorrowDetailsRepository.update()]: ' + res.get('error')
            self.logger.error(error)

        self.post_update(borrow_details)

    # def handle_event(self, event:Union[Event, None]=None):
    #     self.must_update = True


class OrderRepository(BaseRepository):

    EVENT_TYPES = [OrderCreatedEvent, OrderFilledEvent, OrderCancelledEvent]

    def __init__(self, client:object=None, ttl:int=30):

        BaseRepository.__init__(self, client, ttl)
        self._register_with_event_bus(OrderRepository.EVENT_TYPES)

        self.data: tuple = tuple()

        self.return_as_exchange_orders: bool = False
        self._reason_for_update = None
        self._last_update: int = 0

        self.client: AbstractPrivateBroker

    @property
    def orders(self) -> Tuple[ExchangeOrder]:
        while self.updating:
            time.sleep(0.05)

        if self.must_update:
            self.update()

        return tuple(build_exchange_order(o) for o in self.data) \
            if self.return_as_exchange_orders else self.data

    # --------------------------------------------------------------------------
    def update(self):
        """Updates our order list from the API.

        As the retrieval of all orders is more 'expensive' and takes
        more time, we try to reduce the request to orders that were
        created or updated after the last order that we already have.
        """
        self.logger.debug(
            f'running update - start time: {unix_to_utc(self._last_update)}'\
                f' ({self._last_update})'
        )
        end = int(time.time())

        # do this if we don't have any orders yet
        if self._last_update == 0:
            start = int(end - 3600 * 24 * 30)
            orders = self._get_orders(start=start, end=end)

        # otherwise only get new orders and orders that changed their
        # status recently
        else:
            end = int(time.time()) * 1_000
            start = int(self._last_update)
            new_orders = self._get_orders(start=None, end=None)

            if new_orders:
                new_order_ids = [o['orderId'] for o in new_orders]

                # remove duplicates and orders that have a new status
                old_orders = [
                    o for o in self.data if o['orderId'] not in new_order_ids
                    ]
                orders = [*old_orders, *new_orders]

            else:
                orders = self.data

        orders = sorted(orders, key=lambda x: x['updateTime'])

        # just to be sure, remove duplicates again from final result
        orders = tuple(
            o for n, o in enumerate(orders) if o not in orders[n + 1:]
        )

        self.post_update(orders)
        self._last_update = int(time.time() * 1000)

    def _get_orders(self, start: int, end: int) -> list:
        res = self.client.get_orders(start=self._last_update, end=end)

        if res.get('success'):
            orders = res.get('message')
            orders = [self._values_to_numeric(item) for item in orders]
            self.logger.debug(f'got {len(orders)} new orders ...')
        else:
            orders = []
            error = '[OrderRepository.update()]: ' + res.get('error')
            self.logger.error(error)

        return orders

    def _get_update_time_from_most_recent_order(self):
        orders = sorted(self.data, key = lambda item: item['updateTime'])
        return (int(orders[-1].get('updateTime') / 1000) - 1)


# ==============================================================================
class PublicRepository:

    def __init__(self, client):
        """Initializes the repository.

        :param client: An exchange client that is derived from the
        abstract Broker classes and implements its methods.
        :type client: object
        :param mode: 'live' or 'paper trading' or 'backtest',
        defaults to 'live'
        :type mode: str, optional
        """

        self.symbols_repository = SymbolRepository(client=client, ttl=24*3600)
        self.tickers_repository = TickerRepository(client=client, ttl=21)

        # add repositories for risk limits and borrow details,
        # but only if we are using one of the margin markets
        try:
            market = client.market
        except:
            market = 'spot'

        if 'margin' in market.lower():
            self.margin_configuration_repository = MarginConfigurationRepository(
                client=client, ttl=24*3600
            )
        else:
            self.margin_configuration_repository = MarginConfigurationRepository(
                client=client, ttl=24*3600
            )

        self._initialize_repositories()

    # --------------------------------------------------------------------------
    @property
    def symbols(self) -> Tuple[dict]:
        return self.symbols_repository.symbols

    @property
    def tickers(self) -> Tuple[dict]:
        return self.tickers_repository.tickers

    @property
    def margin_configuration(self):
        return self.margin_configuration_repository.margin_configuration \
            if self.margin_configuration_repository else None

    # --------------------------------------------------------------------------
    def notify(self, event:Event):
        event_bus.publish_event(event)

    # --------------------------------------------------------------------------
    def _initialize_repositories(self):
        """Runs an initial update for all repositories that don't
        change very often.

        By doing that we can insure that all calls to these
        repositories can be answered from the cache right from
        the start (=much faster)
        """

        repositories = [self.symbols_repository, self.tickers_repository]

        if self.margin_configuration_repository:
            repositories.append(self.margin_configuration_repository)

        # ......................................................................
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
            [e.submit(r.update) for r in repositories]

    async def __async_initialize(self, loop, executor, repositories):
        """Updates the 'static' repositories asynchronously.

        :param loop: the event loop
        :type loop: async event loop
        :param executor: executor for the methods that are called
        :type executor: ThreadPoolExecutor
        :param repositories: repositories to update
        :type repositories: BaseRepository
        """
        tasks = [
            loop.run_in_executor(executor, r.update,) for r in repositories
            ]

        await asyncio.gather(*tasks)


class PrivateRepository:

    def __init__(self, client):
        """Initializes the repository.

        :param client: An exchange client that is derived from the
        abstract Broker classes and implements its methods.
        :type client: object
        :param mode: 'live' or 'paper trading' or 'backtest',
        defaults to 'live'
        :type mode: str, optional
        """

        self.account_repository = AccountRepository(client=client, ttl=25)
        self.orders_repository = OrderRepository(client=client, ttl=25)

        # add repositories for risk limits and borrow details,
        # but only if we are using one of the margin markets
        market = client.market if client.market else 'spot'

        if 'margin' in market.lower():
            self.risk_limits_repository = RiskLimitRepository(
                client=client, ttl=24*3600
            )
            self.borrow_details_repository = BorrowDetailsRepository(
                client=client, ttl=25
            )
        else:
            self.risk_limits_repository = None
            self.borrow_details_repository = None

        try:
            self._initialize_repositories()
        except Exception as e:
            logger.debug(e)

    # --------------------------------------------------------------------------
    @property
    def account(self):
        return self.account_repository.account

    @property
    def debt_ratio(self):
        return self.account_repository.debt_ratio

    @property
    def orders(self):
        return self.orders_repository.orders

    @property
    def risk_limits(self):
        return self.risk_limits_repository.risk_limits \
            if self.risk_limits_repository else None

    @property
    def borrow_details(self):
        return self.borrow_details_repository.borrow_details \
            if self.borrow_details_repository else None

    # --------------------------------------------------------------------------
    def notify(self, event:Event):
        event_bus.publish_event(event)

    # --------------------------------------------------------------------------
    def _initialize_repositories(self):
        """Runs an initial update for all repositories.

        By doing that we can insure that all calls to these
        repositories can be answered from the cache right from
        the start (=much faster)
        """

        repositories = [self.account_repository, self.orders_repository]

        if self.risk_limits_repository:
            repositories.append(self.risk_limits_repository)
        if self.borrow_details_repository:
            repositories.append(self.borrow_details_repository)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
            [e.submit(r.update) for r in repositories]


