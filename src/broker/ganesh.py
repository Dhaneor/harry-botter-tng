#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""
import time
import math
import logging

from typing import Union, List, Tuple

from src.exchange.exchange_interface import (IExchangePublic, IExchangeTrading,
                                             IExchangeMargin)
from src.exchange.util.exchange_factory import get_exchange
from src.broker.models.orders import (MarketOrder, LimitOrder, Order, StopLimitOrder,
                           StopMarketOrder)
from src.broker.models.exchange_order import ExchangeOrder, build_exchange_order
from src.broker.models.symbol import SymbolFactory, Symbol
from src.broker.models.margin_loan_info import MarginLoanInfo
from data_sources.websockets.ws_kucoin import KucoinWebsocketPrivate

logginda = logging.getLogger('main.Ganesh')
logginda.setLevel(logging.DEBUG)

VALID_MARKETS = ['spot', 'cross margin']
ORDER_OBJECTS = True

# =============================================================================
"""The properties in the classes below (which are used for the composition
of the Ganesh class) allow using cached versions for some frequently
used informations. These cached infos are held in different repositories
that update automatically after some time (specified in the init params)
or can be force updated if necessary.

This saves time by avoiding API  calls for informations that do not
change very often.
"""

class GaneshExchange:
    """This class handles access to all PUBLIC methods on the exchange.

    The public methods (API endpoints) allow retrieval of general
    informations like symbol information, exchange status, etc.
    """
    def __init__(self, exchange: str, market: str):

        self.symbol_factory = SymbolFactory(
            exchange=exchange, market=market
            )

        self.exchange: IExchangePublic
        self.market: str

    # --------------------------------------------------------------------------
    @property
    def server_time(self) -> Union[int, None]:
        return None if not (server_time := self.exchange.get_server_time()) \
            else server_time

    @property
    def system_status(self) -> dict:
        res = self.exchange.get_server_status()

        if res and res['success']:
            res['message']['latency'] = res.get('execution time', -1)
            return res['message']

        return {'msg' : 'no connection to host',
                'status' : 'unknown'
                }

    @property
    def we_have_a_connection(self) -> bool:
        return self.system_status.get('status', 'unkknown') == 'open'


    @property
    def symbols(self) -> Tuple[dict]:
        """Gets a list of all symbols on the exchange.

        :return: all symbols available on the exchange, filtered when
        the 'market' is CROSS MARGIN or ISOLATED MARGIN
        :rtype: List[dict]
        """
        symbols = self.exchange.get_symbols()

        if self.market.lower() == 'SPOT':
            return tuple(symbols)
        else:
            return tuple(s for s in symbols if 'MARGIN' in s['permissions'])

    @property
    def valid_symbols(self) -> Tuple[str]:
        return tuple(item['symbol'] for item in self.symbols)

    @property
    def tickers(self) -> Tuple[dict]:
        return self.exchange.get_all_tickers()

    @property
    def valid_assets(self) -> Tuple[str]:
        """Gets all valid assets for the exchange.

        The assets are automatically filtered by selfsymbols if the
        current market is CROSS MARGIN or ISOLATED MARGIN.

        :return: all valid assets
        :rtype: Tuple[str]
        """
        return tuple(set([sym['baseAsset'] for sym in self.symbols]))


    @property
    def valid_quote_assets(self) -> Tuple[str] :
        return tuple(set([sym['quoteAsset'] for sym in self.symbols]))


    @property
    def currencies(self) -> List[dict]:
        return self.exchange.get_currencies()


    # --------------------------------------------------------------------------
    def get_symbol(self, symbol_name: Union[str, None]=None,
                   base_asset:Union[str, None]=None,
                   quote_asset:Union[str, None]=None
                   ) -> Union[Symbol, None]:

        if all(arg is None for arg in(symbol_name, base_asset, quote_asset)):
            raise ValueError(
                'neither symbol_name nor base_asset+quote_asset were given'
                )

        if symbol_name:
            if symbol_name in self.valid_symbols:
                return self.__build_symbol_object(
                    self.exchange.get_symbol(symbol_name) # type: ignore
                    )
            else:
                raise ValueError(f'{symbol_name} is not a valid symbol_name')

        else:
            if not base_asset in self.valid_assets:
                raise ValueError(f'{base_asset} is not a valid base asset')

            if not quote_asset in self.valid_quote_assets:
                raise ValueError(f'{quote_asset} is not a valid quote asset')

            return next(
                filter(
                    lambda x: x.base_asset == base_asset,
                    self.get_all_symbols(quote_asset=quote_asset)
                ),
                None
            )

    def get_all_symbols(self, quote_asset:Union[str, None]=None) -> Tuple[Symbol]:

        all_symbols = self.symbols

        if quote_asset:
            if quote_asset in self.valid_quote_assets:
                all_symbols = [sym for sym in all_symbols \
                    if sym.get('quoteAsset') == quote_asset]
            else:
                raise ValueError(f'{quote_asset} is not a valid quote asset!')

        return tuple(self.__build_symbol_object(sym) for sym in all_symbols)

    def __build_symbol_object(self, symbol_dict: dict) -> Symbol:
        return self.symbol_factory.build_symbol_from_api_response(symbol_dict)


    # --------------------------------------------------------------------------
    def _check_latency(self):
        st = time.time()
        self.exchange.get_server_status()
        return round((time.time() - st) * 1000)


class GaneshAccount:
    """This class handles all ACCOUNT related methods on the exchange."""
    def __init__(self):
        self.exchange: Union[IExchangeTrading, IExchangeMargin]
        self.market: str

    # --------------------------------------------------------------------------
    @property
    def account(self) -> Union[Tuple[dict], None]:
        return self.exchange.get_account()

    @property
    def valid_assets(self) -> Union[Tuple[str], None]:
        return tuple(i['asset'] for i in self.account) if self.account else None

    @property
    def debt_ratio(self) -> Union[float, None]:
        return 1 if self.market == 'spot' else self.exchange.get_debt_ratio() #type:ignore

    # --------------------------------------------------------------------------
    def get_balance(self, asset: str) -> Union[dict, None]:
        """Get the balance for one asset.

        :param asset: a currency/asset, 'BTC' for instance
        :type asset: str
        :return: API result, balance for asset
        :rtype: dict

        .. code:: python

            {
                'asset': 'XLM',
                'borrowed': '0',
                'free': '0',
                'locked': '0',
                'total': '0'},

            }
        """
        if self.account:
            return next(
                filter(lambda x: x['asset'] == asset, self.account),
                None
            )


class GaneshOrders:
    """This class handles all ORDERS related methods on the exchange.

    The class can retrieve all kinds of informations about current or
    past orders. It also takes care of the execution of orders
    including the handling of (some) error messages from the exchange.
    """
    def __init__(self):
        self.exchange: IExchangeTrading
        self.max_retries = 3

    @property
    def orders(self) -> Union[Tuple[ExchangeOrder], None]:
        if orders := self.exchange.get_orders():
            return tuple(map(build_exchange_order, orders))

    # --------------------------------------------------------------------------
    def get_all_active_orders(self,
                               symbol: Union[str, None]=None,
                               side:Union[str, None]=None
                               ) -> Union[Tuple[ExchangeOrder], None]:
        if orders := self.exchange.get_active_orders(symbol=symbol, side=side):
            return tuple(build_exchange_order(order) for order in orders)

    def get_active_stop_orders(self, symbol: Union[str, None]=None,
                               ) -> Union[Tuple[ExchangeOrder], None]:
        if orders := self.exchange.get_active_stop_orders(symbol):
            return tuple(map(build_exchange_order, orders))

    def get_order(self, order_id:str) -> Union[ExchangeOrder, None]:
        if orders := self.orders:
            return next(
                filter(lambda x: x.order_id == order_id, orders),
                None
            )

    # -------------------------------------------------------------------------
    # all the methods to set and delete orders
    #
    # use execute()  as general entry point!
    def execute(self, order: Order, detailed_result: bool=True) -> Order:
        """Execute an order.

        :param order: order to be excuted
        :type order: Order object
        :return: the (hopefully) executed order (with result)
        :rtype: Order
        """
        executed, counter, response = False, 0, {}

        while not executed and counter < self.max_retries:
            try:
                if isinstance(order, StopMarketOrder):
                    response = self._stop_loss_market(order)

                elif isinstance(order, MarketOrder):
                    if order.side == 'BUY':

                        if not order.auto_borrow:
                            if self._check_if_we_need_to_borrow(order):
                                order.auto_borrow = True

                        if order.auto_borrow and not order.quote_qty:
                            self._change_base_qty_to_quote_qty(order)

                        response = self._market_buy(order)

                    elif order.side == 'SELL':

                        if not order.auto_borrow:
                            if self._check_if_we_need_to_borrow(order):
                                order.auto_borrow = True

                        if order.auto_borrow and not order.base_qty:
                            self._change_quote_qty_to_base_qty(order)

                        response = self._market_sell(order)

                elif isinstance(order, StopLimitOrder):
                    response = self._stop_loss_limit(order)

                elif isinstance(order, LimitOrder):
                    if order.side == 'BUY':
                        if self._check_if_we_need_to_borrow(order):
                            order.auto_borrow = True
                        response = self._limit_buy(order)

                    elif order.side == 'SELL':
                        if self._check_if_we_need_to_borrow(order):
                            order.auto_borrow = True
                        response = self._limit_sell(order)

                else:
                    raise ValueError(f'invalid order type: {type(order)}')

                # -----------------------------------------------------------------
                if isinstance(
                    order, (LimitOrder, StopMarketOrder, StopLimitOrder)):
                    order.status = 'CREATED'
                elif isinstance(order, MarketOrder):
                    order.status = 'FILLED'

                if response:
                    order.order_id = response.get('orderId', 'unknown')

                return order

            except ValueError as e:
                logginda.error(e)

            except Exception as e:
                counter += 1
                order.execution_errors.append(e)
                if 'funds' in str(e):
                    break

        order.status = 'FAILED'
        return order

    # -------------------------------------------------------------------------
    def _market_buy(self, order: MarketOrder) -> Union[dict, None]:
        return self.exchange.buy_market(
            client_order_id=order.client_order_id,
            symbol=order.symbol.name,
            base_qty=order.base_qty,
            quote_qty=order.quote_qty,
            auto_borrow=order.auto_borrow
        )

    def _market_sell(self, order: MarketOrder) -> Union[dict, None]:
        return self.exchange.sell_market(
            client_order_id=order.client_order_id,
            symbol=order.symbol.name,
            base_qty=order.base_qty,
            auto_borrow=order.auto_borrow
        )

    def _limit_buy(self, order: LimitOrder) -> Union[dict, None]:
        return self.exchange.buy_limit(
            client_order_id=order.client_order_id,
            symbol=order.symbol.name,
            base_qty=str(order.base_qty) if order.base_qty else None,
            price=str(order.limit_price),
            auto_borrow=order.auto_borrow
        )

    def _limit_sell(self, order: LimitOrder) -> Union[dict, None]:
        return self.exchange.sell_limit(
            client_order_id=order.client_order_id,
            symbol=order.symbol.name,
            base_qty=str(order.base_qty) if order.base_qty else None,
            price=str(order.limit_price),
            auto_borrow=order.auto_borrow
        )

    def _stop_loss_limit(self, order: StopLimitOrder) -> Union[dict, None]:
        return self.exchange.stop_limit(
            symbol=order.symbol.name,
            side=order.side,
            base_qty=str(order.base_qty),
            stop_price=str(order.stop_price),
            limit_price=str(order.limit_price),
            client_order_id=order.client_order_id
        )

    def _stop_loss_market(self, order: StopMarketOrder) -> Union[dict, None]:

        return self.exchange.stop_market(
            symbol=order.symbol.name,
            side=order.side,
            base_qty=str(order.base_qty),
            stop_price=str(order.stop_price),
            client_order_id=order.client_order_id
        )

    # -------------------------------------------------------------------------
    # cancel an order by a given order id
    def cancel(self, order_id: str, ) -> dict:
        return self.exchange.cancel_order( order_id=order_id)

    def cancel_all(self, symbol:str) -> Union[dict, None]:
        return self.exchange.cancel_all_orders(symbol=symbol)

    def _check_if_we_need_to_borrow(self, order: Order) -> bool:
        symbol = order.symbol

        if order.side == 'BUY':
            balance = self.get_balance(symbol.quote_asset) # type: ignore

            if order.base_qty and order.last_price:
                needed = order.base_qty * order.last_price
            elif order.quote_qty:
                needed = order.quote_qty
            else:
                raise Exception(f'invalid order: {order}')

            available = balance.get('free')

        else:
            balance = self.get_balance(symbol.base_asset) # type: ignore
            needed = order.base_qty
            available = balance.get('free')

        return needed > available

    def _change_base_qty_to_quote_qty(self, order: Order) -> Order:

        precision = order.symbol.quote_precision # type: ignore
        quote_qty = order.base_qty * order.last_price # type: ignore

        order.quote_qty = math.floor(
                    quote_qty * 10**precision) / 10**precision

        order.base_qty = None

        return order

    def _change_quote_qty_to_base_qty(self, order:Order) -> Order:

        precision = order.symbol.base_precision # type: ignore
        base_qty = order.quote_qty * order.last_price # type: ignore

        order.base_qty = math.floor(
                    base_qty * 10**precision) / 10**precision

        order.quote_qty = None

        return order


class GaneshLoans:
    """This class handles all (margin) LOANS related methods on the
    exchange.
    """
    def __init__(self):
        self.exchange: IExchangeMargin

    @property
    def margin_risk_limit(self):
        res = self.exchange.get_margin_risk_limit()
        return res['message'] if res and res['success'] else None


    @property
    def margin_configuration(self):
        res = self.exchange.get_margin_config()
        return res.get('message') if (res and res.get('success')) else {}


    @property
    def borrow_details(self) -> Union[List[dict], None]:
        res = self.exchange.get_borrow_details_for_all()
        return res.get('message') if (res and res.get('success')) else None

    # -------------------------------------------------------------------------
    def get_margin_loan_info(self, asset:str) -> Union[MarginLoanInfo, None]:

        config = self.margin_configuration

        if not config:
            raise Exception('unable to get margin configuration')

        valid_margin_assets = config.get('currencyList', [])

        if not asset in valid_margin_assets:
            raise ValueError(f'{asset} is not a valid margin asset')

        risk_limit = self.get_risk_limit_for_asset(asset)
        details = self.get_borrow_details_for_asset(asset)

        if config and risk_limit and details:
            return MarginLoanInfo(
                asset=asset,
                currency=risk_limit.get('currency', ''),
                precision=risk_limit.get('precision', 0),
                borrow_max_amount=risk_limit.get('borrowMaxAmount', 0),
                buy_max_amount=risk_limit.get('buyMaxAmount', 0),
                hold_max_amount=risk_limit.get('holdMaxAmount', 0),
                available_balance=details.get('availableBalance', 0),
                hold_balance=details.get('holdBalance', 0),
                total_balance=details.get('totalBalance', 0),
                liability=details.get('liability', 0),
                max_borrow_size=details.get('maxBorrowSize', 0),
                max_leverage=config.get('maxLeverage', 1),
                liq_debt_ratio=float(config.get('liqDebtRatio', 0.97)),
                warning_debt_ratio=float(config.get('warningDebtRatio', 0.95))
        )

    def borrow(self, asset:str, amount:float) -> dict:
        return self.exchange.borrow(  # type: ignore
            currency=asset, size=amount, type='FOK'
        )

    def repay(self, asset:str, amount:float) -> dict:
        return self.exchange.repay(currency=asset, size=amount) # type: ignore

    # -------------------------------------------------------------------------
    def get_risk_limit_for_asset(self, asset: str) -> dict:
        if self.margin_risk_limit:
            res = [
                item for item in self.margin_risk_limit \
                if item.get('currency') == asset
            ]

            if res:
                return res[0]
            else:
                raise ValueError(f'asset not found in risk limits! ({asset})')
        else:
            raise Exception('unable to get margin risk limits')

    def get_borrow_details_for_asset(self, asset:str) -> dict:
        return self.exchange.get_borrow_details(asset) # type: ignore

    def get_liability_for_asset(self, asset: Union[str, None]=None):
        return self.exchange.get_liability(asset) # type: ignore


class GaneshFees:
    """This class provides informations about fees for all assets"""

    def __init__(self):
        self.exchange: IExchangeTrading

    def get_fees(self) -> List[dict]:
        return self.exchange.get_fees()


# ==============================================================================
class Ganesh(
    GaneshExchange, GaneshAccount, GaneshOrders, GaneshLoans, GaneshFees):
    '''This class handles all necessary calls to the API.

    You could also say that Ganesh is our 'broker' who gets things done
    for us on the exchange.
    '''

    def __init__(self, exchange:str, market:str, credentials:dict):
        """Ganesh initialization.

        :param exchange: the name of the exchange to be used
        :type exchange: str
        :param market: 'SPOT' or 'CROSS MARGIN'
        :type market: str
        :param credentials: API key, API secret, ...
        :type credentials: dict
        :raises ValueError: ValueError if exchange/market is None or no
                            valid exchange name/market type was provided
        """
        self.name = 'GANESH'
        self.function = 'Junior Execution Manager'

        self.exchange = get_exchange( # type:ignore
            exchange=exchange, market=market, credentials=credentials
        )

        self.market = market.lower()
        self._max_retries = 2
        self._latencies: list = []

        self.credentials = credentials

        GaneshExchange.__init__(self, exchange, market)
        GaneshAccount.__init__(self)
        GaneshOrders.__init__(self)
        GaneshLoans.__init__(self)


    def __repr__(self) -> str:
        return f'{self.name.upper()} ({self.function}) '

    @property
    def repository(self):
        return self.exchange.repository

    # -------------------------------------------------------------------------
    # helper methods
    def get_repository(self) -> object:
        return self.exchange.repository

    def get_last_price(self, symbol_name):
        tickers = self.exchange.get_all_tickers()
        t = [t for t in tickers if t['symbol'] == symbol_name]
        if t:
            return float(t[0]['last'])
        else:
            raise ValueError(f'{symbol_name} is not a valid symbol')

    def get_ws_client(self):
        if self.exchange == 'kucoin':
            return KucoinWebsocketPrivate(credentials=self.credentials)
