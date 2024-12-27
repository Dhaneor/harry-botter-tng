#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 23:54:28 2022

@author: dhaneor
"""
import os
import sys
import concurrent.futures
import pandas as pd
import logging
from cachetools import cached, TTLCache

from typing import Tuple, Union, Optional
from uuid import uuid1
from time import time, sleep
from pprint import pprint

# ------------------------------------------------------------------------------
# add the parent directory to the search path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------
from data.rawi.kucoin.kucoin.client import (  # noqa: F401, E402
    Market, Margin, User, Trade
)
from exchange.exchange_interface import (  # noqa: F401, E402
    IExchangePublic,
    IExchangeTrading,
    IExchangeMargin,
)
from exchange.util.ohlcv_download_prepper import (  # noqa: F401, E402
    OhlcvDownloadPrepper
)
from exchange.util.kucoin_order_downloader import (  # noqa: F401, E402
    KucoinOrderDownloader
)
from util.timeops import (  # noqa: F401, E402
    utc_to_unix, interval_to_milliseconds
)
from util.accounting import Accounting  # noqa: F401, E402

logger = logging.getLogger("main.kucoin")
logger.setLevel(logging.INFO)

try:
    from broker.config import CREDENTIALS
except Exception:
    logger.warning("unable to import CREDENTIALS from broker.config")

global DELAY
DELAY = 0
MAX_RETRY = 3
MAX_OHLCV_INTERVALS = 1499
EXCHANGE_OPENED = "September 01, 2017 00:00:00"


# ==============================================================================
def is_exchange_closed():
    client = Market()
    status = client.get_server_status()
    if status:
        return status.get("status", "close") == "close"


def wrap_call(func):
    def inner(*args, **kwargs):
        st = time()
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "400" in str(e):
                    logger.warning(f"{func.__name__} bad request ({args}, {kwargs})")
                    return 400
                elif "401" in str(e):
                    logger.warning(f"{func.__name__} unauthorized")
                    return 401
                elif "403" in str(e):
                    logger.warning(f"{func.__name__} forbidden")
                    return 403
                elif "404" in str(e):
                    logger.warning(f"{func.__name__} not found")
                    return 404
                elif "405" in str(e):
                    logger.warning(f"{func.__name__} method not allowed")
                    return 405
                elif "415" in str(e):
                    logger.error(f"{func.__name__} unsupported media type")
                    return 415
                elif "429" in str(e):
                    logger.warning(f"{func.__name__} hit request rate limit")
                    sleep(3)
                elif "500" in str(e):
                    logger.error(f"{func.__name__} hit internal server error")
                    return 500
                elif "503" in str(e):
                    logger.error(f"{func.__name__} hit service unavailable")
                    return 503
                else:
                    logger.error(e)
            finally:
                logger.debug(f"{func.__name__} took {(time() - st)*1000:.0f} ms")

    return inner


# =============================================================================
class KucoinResponseFormatter:
    # -------------------------------------------------------------------------
    # methods to standardize the response from the Kucoin API to the same
    # format that is given by the Binance API.
    #
    # I started with this bot on  Binance so all the calling methods expect
    # an answer in this format, it also makes more sense to me and as we
    # need a standardized format anyways (if we want to use multiple
    # exchanges), I decided to go with the Binance format.
    def _standardize_ticker(self, ticker: Union[dict, None]) -> Union[dict, None]:
        """Convert ticker response to standard format

        :param ticker: Binance API response for a single ticker
        :type ticker: dict

        .. code:: python
            {
                'averagePrice': '0.00002013',
                'buy': '0.00002035',
                'changePrice': '0.00000106',
                'changeRate': '0.0549',
                'high': '0.00002122',
                'last': '0.00002034',
                'low': '0.00001927',
                'makerCoefficient': '1',
                'makerFeeRate': '0.001',
                'sell': '0.00002037',
                'symbol': 'XRP-BTC',
                'takerCoefficient': '1',
                'takerFeeRate': '0.001',
                'time': 1645304938256,
                'vol': '4314129.61233997',
                'volValue': '87.488393732489632'
            }

        :return: Standardized ticker response
        :rtype: dict
        """
        if not ticker:
            return

        def _get_prec(price):
            try:
                return len(price.split(".")[1])
            except IndexError:
                return 0

        _precision = max(
            _get_prec(ticker["price"]),
            _get_prec(ticker["high"]),
            _get_prec(ticker["low"]),
        )

        _open = float(ticker["last"]) / (1 + float(ticker["changeRate"]))
        _open = str(Accounting.round(amount=_open, ndigits=_precision))  # type:ignore

        return {
            "symbol": ticker["symbol"],
            "ask": ticker["sell"],
            "bid": ticker["buy"],
            "last": ticker["price"],
            "open": _open,
            "high": ticker["high"],
            "low": ticker["low"],
            "priceChange": ticker["changePrice"],
            "priceChangePercent": ticker["changeRate"],
            "volume": ticker["vol"],
            "quoteVolume": ticker["volValue"],
        }

    def _standardize_fill_response(self, fills: list) -> list[dict]:
        """Sandardize fill resonse

        This method transforms a Kucoin fill response to the format that
        Binance uses.

        :param fill: fill information from Kucoin
        :type fill: dict

        This is what we get from Kucoin:

        .. code :: python

            {'counterOrderId': '620848e0fec9a60001ec81f0',
             'createdAt': 1644710113000,
             'fee': '0.000499999995',
             'feeCurrency': 'USDT',
             'feeRate': '0.001',
             'forceTaker': True,
             'funds': '0.499999995',
             'liquidity': 'taker',
             'orderId': '620848e0fd31e50001a59db2',
             'price': '0.822',
             'side': 'buy',
             'size': '0.6082725',
             'stop': '',
             'symbol': 'XRP-USDT',
             'tradeId': '620848e02e113d325d83e24b',
             'tradeType': 'MARGIN_TRADE',
             'type': 'market'
            }

        :returns: dict

        .. code :: python

            {'commission': '0.01092400',
            'commissionAsset': 'USDT',
            'price': '1.09240000',
            'qty': '10.00000000',
            'tradeId': 3246868
            }
        """

        return [
            {
                "commission": fill["fee"],
                "commissionAsset": fill["feeCurrency"],
                "price": fill["price"],
                "qty": fill["size"],
                "tradeId": fill["tradeId"],
            }
            for fill in fills
        ]

    def _standardize_order_response(self, response: dict, with_fills=False) -> dict:
        """Standardize the response format for order responses

        This method transforms Kucoin order responses to the format that
        Binance uses. The Binance format makes more sense to me and is
        better readable

        :param response: the original order response from Kucoin
        :type response: dict

        .. code :: python

            {'cancelAfter': 0,
            'cancelExist': False,
            'channel': 'API',
            'clientOid': '8e13a71e-8de0-11ec-bb49-1e00623eee81',
            'createdAt': 1644875614876,
            'dealFunds': '9.9999999990416',
            'dealSize': '47.8283536',
            'fee': '0.0099999999990416',
            'feeCurrency': 'USDT',
            'funds': '10',
            'hidden': False,
            'iceberg': False,
            'id': '620acf5e19673d0001599fa5',
            'isActive': False,
            'opType': 'DEAL',
            'postOnly': False,
            'price': '0',
            'remark': None,
            'side': 'buy',
            'size': '0',
            'stop': '',
            'stopPrice': '0',
            'stopTriggered': False,
            'stp': '',
            'symbol': 'XLM-USDT',
            'tags': None,
            'timeInForce': 'GTC',
            'tradeType': 'MARGIN_TRADE',
            'type': 'market',
            'visibleSize': '0'
            }
        """
        fills, update_time = [], int(response["createdAt"])

        # determine order status
        if not response["isActive"]:
            is_working = False
            if response["dealSize"] == "0":
                status = "CANCELED"
            else:
                status = "FILLED"

        elif response["isActive"]:
            status, is_working = "NEW", True

        else:
            status, is_working = "UNKMOWN", False

        # if necessary, get the fills for this order from the API and
        # transform these to our standardized format.
        if status == "FILLED" and response["dealSize"] != "0" and with_fills:

            res = super().trading.get_fills(  # type:ignore
                order_id=response["id"],
                start=int(update_time - 60000),
                end=int(update_time + 60000),
            )

            if res["success"]:
                fills_src = res["message"]["items"]

                if fills_src:
                    fills = self._standardize_fill_response(fills=fills_src)

                    try:
                        update_time = int(fills_src[-1].get("createdAt"))
                    except Exception as e:
                        pprint(e)
                        pprint(fills)
            else:
                pprint(res)
                print("unabale to get fills")

        # if it was a stop-loss or stop-entry order, set the correct type
        limit = True if response["type"] == "limit" else False

        if response["stop"] == "loss":
            order_type = "STOP_LOSS_LIMIT" if limit else "STOP_LOSS_MARKET"
        elif response["stop"] == "entry":
            order_type = "STOP_ENTRY_LIMIT" if limit else "STOP_ENTRY_MARKET"
        else:
            order_type = response["type"].upper()

        # return the dictionary in our standardized format, populated with
        # the values that we extracted from the response
        return {
            "clientOrderId": response["clientOid"],
            "cummulativeQuoteQty": response["dealFunds"],
            "executedQty": response["dealSize"],
            "fills": fills,
            "icebergQty": "0.00000000",
            "isWorking": is_working,
            "orderId": response["id"],
            "orderListId": -1,
            "origQty": response["size"],
            "origQuoteOrderQty": response["funds"],
            "price": response["price"],
            "side": response["side"].upper(),
            "status": status,
            "stopPrice": response["stopPrice"],
            "symbol": response["symbol"],
            "time": int(response["createdAt"]),
            "timeInForce": response["timeInForce"],
            "type": order_type,
            "updateTime": update_time,
        }

    def _standardize_stop_order_response(self, response: dict) -> dict:
        """Standardize the response that from the endpoint for stop orders

        :param response: dictionary with the raw response
        :type response: dict

        This is what we get from Kucoin:

        .. code:: python
            {
                'cancelAfter': 0,
                'channel': 'ANDROID',
                'clientOid': None,
                'createdAt': 1645283978114,
                'domainId': 'kucoin',
                'feeCurrency': 'USDT',
                'funds': None,
                'hidden': False,
                'iceberg': False,
                'id': 'vs8nqogh1a5c119d000s0gec',
                'makerFeeRate': '0.00100000000000000000',
                'orderTime': 1645283978114000034,
                'postOnly': False,
                'price': '0.18000000000000000000',
                'remark': None,
                'side': 'sell',
                'size': '23.91430000000000000000',
                'status': 'NEW',
                'stop': 'loss',
                'stopPrice': '0.18100000000000000000',
                'stopTriggerTime': None,
                'stp': None,
                'symbol': 'XLM-USDT',
                'tags': None,
                'takerFeeRate': '0.00100000000000000000',
                'timeInForce': 'GTC',
                'tradeSource': 'USER',
                'tradeType': 'MARGIN_TRADE',
                'type': 'limit',
                'userId': '5fd10f949910b40006395f9e',
                'visibleSize': None
            }

        :returns: the same response in standardized format
        :rtype: dict
        """
        r = response
        try:
            orig_quote_qty = float(r["size"]) * float(r["price"])
        except Exception:
            orig_quote_qty = 0

        # determine standardized order type
        if r["stop"] == "loss":
            type = "STOP_LOSS_LIMIT" if r["type"] == "limit" else "STOP_LOSS_MARKET"
        # Binance doesn't have stop-entry orders, so this is just
        # for being complete and to prevent errors. this will
        # probably never be used in production mode
        elif r["stop"] == "entry":
            type = "STOP_ENTRY_LIMIT" if r["type"] == "limit" else "STOP_ENTRY_MARKET"
        else:
            type = "UNKNOWN"

        return {
            "clientOrderId": r["clientOid"],
            "cummulativeQuoteQty": "0",
            "executedQty": "0",
            "fills": [],
            "icebergQty": "0.00000000",
            "isWorking": True if r["status"] == "NEW" else False,
            "orderId": r["id"],
            "orderListId": -1,
            "origQty": r["size"],
            "origQuoteOrderQty": orig_quote_qty,
            "price": r["price"],
            "side": r["side"].upper(),
            "status": r["status"],
            "stopPrice": r["stopPrice"],
            "symbol": r["symbol"],
            "time": int(r["orderTime"] / 1_000_000),
            "timeInForce": r["timeInForce"],
            "type": type,
            "updateTime": int(r["orderTime"] / 1_000_000),
        }

    # ..........................................................................
    def _standardize_symbol_information(self, symbol_info: dict) -> dict:
        """Standardizes the symbol information to Binance format.

        :param symbol_info: the information for one symbol
        :type symbol_info:  dict

        .. code: python

            {
                'baseCurrency': 'XLM',
                'baseIncrement': '0.0001',
                'baseMaxSize': '10000000000',
                'baseMinSize': '0.1',
                'enableTrading': True,
                'feeCurrency': 'USDT',
                'isMarginEnabled': True,
                'market': 'USDS',
                'minFunds': '0.1',
                'name': 'XLM-USDT',
                'priceIncrement': '0.000001',
                'priceLimitRate': '0.1',
                'quoteCurrency': 'USDT',
                'quoteIncrement': '0.000001',
                'quoteMaxSize': '99999999',
                'quoteMinSize': '0.01',
                'symbol': 'XLM-USDT'
            }

        :returns:   symbol_info in Binance format
        :rtype: dict

        .. code:: python

            {
                "symbol": "ETHBTC",
                "status": "TRADING",
                "baseAsset": "ETH",
                "baseAssetPrecision": 8,
                "quoteAsset": "BTC",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "baseCommissionPrecision" : 8,
                "quoteCommissionPrecision" : 8,
                "orderTypes": [
                    "LIMIT",
                    "LIMIT_MAKER",
                    "MARKET",
                    "STOP_LOSS",
                    "STOP_LOSS_LIMIT",
                    "TAKE_PROFIT",
                    "TAKE_PROFIT_LIMIT"
                ],
                "icebergAllowed": true,
                "ocoAllowed": true,
                "isSpotTradingAllowed": true,
                "isMarginTradingAllowed": true,
                "filters": [
                    //These are defined in the Filters section.
                    //All filters are optional
                ],
                "permissions": [
                    "SPOT",
                    "MARGIN"
                ]
            }
        """
        s = symbol_info

        base_asset_precision = len(s["baseIncrement"].split(".")[1])
        quote_asset_precision = len(s["quoteIncrement"].split(".")[1])

        spot_enabled, margin_enabled = s["enableTrading"], s["isMarginEnabled"]

        permissions = ["SPOT"] if spot_enabled else []
        if s["isMarginEnabled"]:
            permissions.append("MARGIN")

        return {
            "symbol": s["symbol"],
            "status": "TRADING" if spot_enabled else "BREAK",
            "baseAsset": s["baseCurrency"],
            "baseAssetPrecision": base_asset_precision,
            "quoteAsset": s["quoteCurrency"],
            "quotePrecision": quote_asset_precision,
            "quoteAssetPrecision": quote_asset_precision,
            "baseCommissionPrecision": base_asset_precision,
            "quoteCommissionPrecision": quote_asset_precision,
            "quoteOrderQtyMarketAllowed": True,
            "orderTypes": [
                "LIMIT",
                "MARKET",
                "STOP_LOSS",
                "STOP_LOSS_LIMIT",
                "TAKE_PROFIT",
                "TAKE_PROFIT_LIMIT",
            ],
            "icebergAllowed": True,
            "ocoAllowed": True,
            "isSpotTradingAllowed": spot_enabled,
            "isMarginTradingAllowed": margin_enabled,
            "filters": self._extract_filters(symbol_info=s),
            "permissions": permissions,
        }

    def _extract_filters(self, symbol_info: dict) -> list:
        """Extract the information that is under 'filters' in Binance
        format from Kucoin symbol info.

        :param symbol_info: the information for one symbol
        :type symbol_info:  dict

        This is what we get from Kucoin:

        .. code: python

            {
                'baseCurrency': 'XLM',
                'baseIncrement': '0.0001',
                'baseMaxSize': '10000000000',
                'baseMinSize': '0.1',
                'enableTrading': True,
                'feeCurrency': 'USDT',
                'isMarginEnabled': True,
                'market': 'USDS',
                'minFunds': '0.1',
                'name': 'XLM-USDT',
                'priceIncrement': '0.000001',
                'priceLimitRate': '0.1',
                'quoteCurrency': 'USDT',
                'quoteIncrement': '0.000001',
                'quoteMaxSize': '99999999',
                'quoteMinSize': '0.01',
                'symbol': 'XLM-USDT'
            }

        :returns: list with dictionaries, that describe the filters
        :rtype: list
        """
        s = symbol_info

        return [
            {
                "filterType": "PRICE_FILTER",
                "maxPrice": "1000000.00000000",
                "minPrice": "0.00000001",
                "tickSize": s["priceIncrement"],
            },
            {
                "avgPriceMins": 5,
                "filterType": "PERCENT_PRICE",  # no way and no need to
                "multiplierDown": "0.2",  # populate this with real
                "multiplierUp": "5",
            },  # values, so only defaults (not used)
            {
                "filterType": "LOT_SIZE",
                "maxQty": s["baseMaxSize"],
                "minQty": s["baseMinSize"],
                "stepSize": s["baseIncrement"],
            },
            {
                "applyToMarket": True,
                "avgPriceMins": 5,
                "filterType": "MIN_NOTIONAL",
                "minNotional": s["quoteMinSize"],
            },
            {"filterType": "ICEBERG_PARTS", "limit": 20},  # always 20 on Kucoin
            {
                "filterType": "MARKET_LOT_SIZE",
                "maxQty": s["baseMaxSize"],
                "minQty": s["baseMinSize"],
                "stepSize": s["baseIncrement"],
            },
            {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
            {"filterType": "MAX_NUM_ALGO_ORDERS", "maxNumAlgoOrders": 10},
        ]

    def _standardize_account(self, item: dict) -> dict:
        """Transform one item (asset) to standardized format.

        :param item: dictionary with the values for one asset in user account
        :type item: dict

        When querying the account, we get a list of dictionaries from
        Kucoin. Each dictionary represents one asset (currency) and has
        the format:

        .. code:: python

            {'availableBalance': '0',
             'currency': 'ZRX',
             'holdBalance': '0',
             'liability': '0',
             'maxBorrowSize': '7300',
             'totalBalance': '0'
             }

        :returns: The converted item as dictionary
        :rtype: dict

        The returned dictionary will have the format:

        .. code:: python
            {'currency' : <name (ticker) of asset/currency>,
             'free' : <amount not locked in orders>,
             'locked' : <amount locked in orders>,
             'borrowed' : <borrowed amount (if any)>,
             'total' : <sum of free and locked>
             }
        """
        return {
            "asset": item["currency"],
            "free": item["availableBalance"],
            "locked": item["holdBalance"],
            "borrowed": item["liability"],
            "total": item["totalBalance"],
        }

    def _standardize_fees(self, fees: dict):
        """Convert fee information to standard format

        :param fees: dictionary with fee information for one symbol
        :type fees: dict

        .. code:: python
            {
             'makerFeeRate': '0.001',
             'symbol': 'XRP-USDT',
             'takerFeeRate': '0.001'
            }

        :returns: dictionary with standardized fee information
        :rtype: dict

        .. code:: python
        {
         'symbol' : <name of symbol>,
         'maker' : <maker fee rate>,
         'taker : <taker fee rate>
        }
        """
        return {
            "symbol": fees["symbol"],
            "maker": fees["makerFeeRate"],
            "taker": fees["takerFeeRate"],
        }


# =============================================================================
class Public(IExchangePublic, OhlcvDownloadPrepper, KucoinResponseFormatter):
    """This class handles all the public calls to the Kucoin API."""

    INTERVALS = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "2h": "2hour",
        "4h": "4hour",
        "8h": "8hour",
        # '12h' : '12hour',
        "1d": "1day",
        "1w": "1week",
    }

    def __init__(self):
        self.market_client: Market = Market()

        self.max_workers = 10  # max number of simultaneous downloads
        self.delay = 0

        self.result = {
            "success": False,
            "message": None,
            "error": None,
            "status_code": None,
        }

    def __enter__(self):
        if not self.market_client:
            self.market_client = Market()
        return self

    def __exit__(self, *args, **kwargs):
        return False

    # -------------------------------------------------------------------------
    # methods to get general information (time, status, markets ...)
    @wrap_call
    def get_server_time(self) -> Union[int, None]:
        return (
            int(s_time)
            if (s_time := self.market_client.get_server_timestamp())
            else None
        )

    @cached(cache=TTLCache(maxsize=4096, ttl=10))
    @wrap_call
    def get_server_status(self) -> Union[dict, None]:
        """Get the current system status

        :return: returns the system status
        :rtype: dict

        .. code:: python
            {
            "status": "open",               // open, close, cancelonly
            "msg":  "upgrade match engine"  // remark for operation
            }
        """
        return self.market_client.get_server_status()

    @cached(cache=TTLCache(maxsize=4096, ttl=3600))
    @wrap_call
    def get_currencies(self):
        return self.market_client.get_currencies()

    @cached(cache=TTLCache(maxsize=4096, ttl=3600))
    @wrap_call
    def get_markets(self):
        return self.market_client.get_market_list()

    @cached(cache=TTLCache(maxsize=4096, ttl=3600))
    @wrap_call
    def get_symbols(self, quote_asset: Union[str, None] = None) -> Tuple[dict]:
        symbols = self.market_client.get_symbol_list()
        if symbols:
            if quote_asset:
                symbols = filter(lambda x: x["quoteCurrency"] == quote_asset, symbols)

            return tuple(self._standardize_symbol_information(s) for s in symbols)
        else:
            return tuple()

    @cached(cache=TTLCache(maxsize=4096, ttl=3600))
    @wrap_call
    def get_symbol(self, symbol: str) -> Union[dict, None]:
        return next(filter(lambda x: x["symbol"] == symbol, self.get_symbols()), None)

    @cached(cache=TTLCache(maxsize=4096, ttl=5))
    @wrap_call
    def get_ticker(self, symbol: str) -> Union[dict, None]:
        return self.market_client.get_ticker(symbol)

    @cached(cache=TTLCache(maxsize=4096, ttl=5))
    @wrap_call
    def get_all_tickers(self) -> Tuple[dict]:
        res = self.market_client.get_all_tickers()
        if res and (tickers := res.get("ticker", [])):
            return tuple(tickers)
        else:
            return tuple()

    # -------------------------------------------------------------------------
    # method to get historical ohlcv data
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Union[int, str, None] = None,
        end: Union[int, str, None] = None,
        as_dataframe: bool = True,
    ) -> dict:
        """Download historical ohlcv data for a symbol.

        The format of the returned dict is as follows:
        {
            "success": True or False,
            "message": <data> or None,
            "error": None or <error>,
            "status_code": 200 or <Kucoin error code>,
        }

        :param symbol: name of the symbol, e.g. 'BTC-USDT'
        :type symbol: str
        :param interval: '1m', '5m', '15m', '30m', '1h', ...
        :type interval: str
        :param start: start from time, defaults to None
        can be a datetime 'January 1, 2018 00:00:00' or
        a unix timestamp in milliseconds, or None - if not
        provided, 1000 periods will be downloaded
        :type start: Union[int, str, None], optional
        :param end:end with time, defaults to None
        same formats as start, if not provided the current
        time will be used
        :type end: Union[int, str, None], optional
        :param as_dataframe:return as dataframe or raw data, defaults to True
        :type as_dataframe: bool, optional
        :return: ohlcv data,
        :rtype: dict
        """
        # srt start and end times, if not provided
        start = -1000 if not start else start
        end = time.time() if not end else end

        # prepare a download request ... Kucoin won't let us
        # download more than 1000 periods, so we need to split
        # the request into chunks ... this happens with the
        # method that is called here.
        try:
            dl_request = self._prepare_request(
                symbol=symbol, start=start, end=end, interval=interval
            )
        except Exception as e:
            logger.error(e)
            return {
                "success": False,
                "message": None,
                "error": f"unable to build download request: {e}",
            }

        logger.debug(f"download request(s): {dl_request}")

        # ......................................................................
        # download the data in parallel with threads
        _results, futures = [], []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            for chunk in dl_request["chunks"]:
                kwargs = {
                    "symbol": dl_request.get("symbol"),
                    "interval": dl_request.get("interval"),
                    "start_ts": chunk[0],
                    "end_ts": chunk[1],
                }

                futures.append(executor.submit(self._fetch_ohlcv, **kwargs))

            futures, _ = concurrent.futures.wait(futures)
            for future in futures:
                _results.append(future.result())

        # extract data, errors and status codes from results

        # Kucoin returns [{'code': 20000, 'data': []}] for missing periods,
        # where probably the exchange was down or in maintenance mode.
        # This is different from the returned list (with no dictionaries in
        # it), which is the normal format of the response.
        # This checks for the returned format ...
        try:
            res = [_res for _res in _results if not isinstance(_res[0], dict)]
        except KeyError:
            return {
                "success": False,
                "message": None,
                "error": "no data available for requested period",
            }
        except TypeError:
            return {"success": False, "message": None, "error": res}
        except Exception as e:
            if "connection" in str(e):
                logger.error("no connection")
            else:
                logger.error(e)

            return {"success": False, "message": None, "error": e}

        # we now have a list of list and need to flatten it
        res = [item for sub_list in res for item in sub_list]

        # convert the result to  a dataframe if parameter as_dataframe is True
        if as_dataframe:
            try:
                res = self._klines_to_dataframe(res)
            except Exception:
                res = pd.DataFrame()

        return {"success": True, "message": res, "arguments": None}

    # ..........................................................................
    @wrap_call
    def _fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Union[dict, None]:
        """This method fetches the klines from the API. It should not
        be called directly because almost always we need to some
        checks before calling the API and as Kucoin delivers 7 days of
        data at most. So, we also need to page  requests that cover
        longer periods. All of this is implemented in  get_ohlcv() and
        this is the method that the caller should use.
        """
        sleep(self.delay)

        return self.market_client.get_kline(
            symbol=symbol,
            kline_type=interval,
            startAt=start_ts,
            endAt=end_ts
        )

    # -------------------------------------------------------------------------
    # helper methods
    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """Converts the raw OHLCV data to a dataframe."""

        # set column names
        columns = [
            "open time",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "quote volume",
        ]

        # build dataframe from list of klines
        df = pd.DataFrame(columns=columns, data=klines).astype(float, copy=False)

        df = df[["open time", "open", "high", "low", "close", "volume", "quote volume"]]

        if df.empty:
            return df

        # convert values to numeric
        df["open time"] = (df["open time"] * 1000).astype(int)

        # add column with human readable 'open time'
        df.insert(1, "human open time", pd.to_datetime(df["open time"], unit="ms"))

        # add column for 'close time' (end of candle)
        interval = df.loc[1, "open time"] - df.loc[0, "open time"] - 1  # type:ignore
        df.insert(6, "close time", (df["open time"] + interval))

        df.sort_values(by="open time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    @wrap_call
    def _get_earliest_valid_timestamp(self, symbol: str, interval: str = "1d") -> int:

        interval_ms = interval_to_milliseconds(interval)
        if not interval_ms:
            raise ValueError(f"{interval} is not a valid interval")

        interval_sec = interval_ms / 1000
        start = int(utc_to_unix(EXCHANGE_OPENED) / 1000)
        end = int(start + MAX_OHLCV_INTERVALS * interval_sec)

        while True:
            res = self._fetch_ohlcv(
                symbol=symbol,
                interval=self.INTERVALS[interval],
                start_ts=start,
                end_ts=0,
            )

            if res:
                return res[-1][0] if isinstance(res, list) else start

            start = end
            end = int(start + 1000 * interval_sec)

    def set_max_workers(self, max_workers: int):
        self.max_workers = max_workers

    def _increase_delay(self, amount: float = 0.5):
        self.delay += amount
        print(f"delay increased to {self.delay}s")

    # NOTE: see client.py for additional methods that we don't need for now ...


class Account(KucoinResponseFormatter):

    def __init__(self, credentials: dict):
        self.user_client: User = User(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )

    # -------------------------------------------------------------------------
    # methods related to the user account
    # @wrap_call
    def get_accounts(
        self, currency: Optional[str] = None, account_type: Optional[str] = None
    ) -> Union[dict, None]:
        return self.user_client.get_account_list(
            currency=currency, account_type=account_type
        )

    @wrap_call
    def get_account_by_id(self, id_: str) -> Union[dict, None]:
        return self.user_client.get_account(accountId=id_)

    @wrap_call
    def create_account(self):
        raise NotImplementedError

    @wrap_call
    def get_account_activity(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # DEPOSIT / WITHDRAW / TRANSFER between accounts
    @wrap_call
    def create_inner_transfer(self):
        raise NotImplementedError

    @wrap_call
    def get_deposits(
        self, currency=None, status=None, start=None, end=None, page=None, limit=None
    ):
        return self.user_client.get_deposit_list(
            currency=currency,
            status=status,
            start=start,
            end=end,
            page=page,
            limit=limit,
        )

    @wrap_call
    def get_withdrawals(
        self,
        currency: Optional[str] = None,
        status: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Union[dict, None]:
        return self.user_client.get_withdrawal_list(
            currency=currency,
            status=status,
            start=start,
            end=end,
            page=page,
            limit=limit,
        )

    @wrap_call
    def get_withdrawal_quotas(self, currency: str) -> Union[dict, None]:
        return self.user_client.get_withdrawal_quota(currency=currency)


class Spot:
    """This class implements additional methods that are specific to
    trade on the SPOT market.

    TODO    implement this class! this is just the skeleton because
            we don't need spot trading for now
    """

    def __init__(self, credentials: dict):
        self.user_client: User = User(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )

        self.trade_client: Trade = Trade(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )


class CrossMargin(IExchangeTrading, IExchangeMargin, KucoinResponseFormatter):
    """This class handles all the private (=require 'Trade' permission) calls
    to the Kucoin API for 'Cross Margin' trading."""

    def __init__(self, credentials: dict):
        self.user_client: User = User(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )

        self.trade_client: Trade = Trade(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )

        self.margin_client: Margin = Margin(
            key=credentials["api_key"],
            secret=credentials["api_secret"],
            passphrase=credentials["api_passphrase"],
        )

        self.account = "margin"
        self.max_retry = 3

    # -------------------------------------------------------------------------
    # MARGIN ACCOUNT related methods
    @wrap_call
    def get_margin_config(self) -> Union[dict, None]:
        return self.margin_client.get_margin_config()

    @wrap_call
    def get_account(self) -> Union[Tuple[dict], None]:
        query_res = self.margin_client.get_margin_account()

        if not query_res:
            return

        return tuple(self._standardize_account(acc) for acc in query_res["accounts"])

    def get_debt_ratio(self) -> Union[float, None]:
        res = self.margin_client.get_margin_account()
        return res["debtRatio"] if res else None

    def get_balance(self, asset: str) -> Union[dict, None]:
        account = self.get_account()

        if account:
            return next(filter(lambda x: x["asset"] == asset, account), None)

    @wrap_call
    def get_fees(self, symbols: Union[list[str], str]) -> Union[list[dict], None]:
        """Gets the fee information for one or more assets.

        TODO:   extend this method or make it a class so we can get
                this data for more than 10 symbols. The Kucoin API
                restricts this query to 10 symbols at the most.

        :param symbols: one or more symbols (max. 10 for now)
        :type symbols: Union[list, str]
        :return: _description_
        :rtype: dict
        """
        if isinstance(symbols, list):
            symbols = ",".join(symbols)

        res = self.user_client.get_actual_fee(symbols=symbols)
        if res:
            return [self._standardize_fees(item) for item in res]
        else:
            return res

    # -------------------------------------------------------------------------
    # QUERYING, CREATING and DELETING ORDERS related
    def get_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        order_type: Optional[str] = None,
        status: Optional[str] = None,
        start: Union[int, str, None] = None,
        end: Union[int, str, None] = None,
    ) -> Union[Tuple[dict], None]:
        """Gets multiple orders, filtered if requested.

        :param symbol: name of the symbol, defaults to None
        :type symbol: str, optional
        :param side: BUY or SELL, defaults to None
        :type side: str, optional
        :param order_type: MARKET, LIMIT, ..., defaults to None
        :type order_type: str, optional
        :param status: NEW, FILLED, CANCELED, defaults to None
        :type status: str, optional
        :param start: start date, defaults to None
        :type start: Union[int, str], optional
        :param end: end date, defaults to None
        :type end: Union[int, str], optional
        :return: list of orders (order: dictionary)
        :rtype: list[dict]
        """

        self.__orders = []
        status = status if status else "done"

        def _add_to_final_result(res):
            if "symbol" in res:
                self.__orders += res["message"]
            else:
                raise Exception(f"something went wrong with the order downloads {res}")

        # ----------------------------------------------------------------------
        # download the data in parallel with threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            futures = []

            kwargs = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "status": status,
                "start": start,
                "end": end,
            }

            futures = [
                executor.submit(self._get_multiple_orders, **kwargs),
                executor.submit(self.get_active_stop_orders, symbol=symbol),
                executor.submit(self.get_active_orders, symbol=symbol),
            ]

            futures, _ = concurrent.futures.wait(futures)

            for future in futures:
                _add_to_final_result(future.result())

        # ......................................................................
        # filter the results based on parameters given
        orders = self.__orders
        if orders:
            # filter for 'side' and 'status' before standardizing format
            if side:
                orders = [o for o in orders if o["side"] == side.upper()]

            if status == "CANCELED":
                orders = [o for o in orders if o["status"] == "CANCELED"]
            elif status == "FILLED":
                orders = [o for o in orders if o["status"] == "FILLED"]

            if order_type:
                orders = [o for o in orders if o["type"] == order_type]

            orders = sorted(orders, key=lambda i: i["updateTime"])

        return tuple(orders)

    def get_order(
        self, order_id: Optional[str] = None, client_order_id: Optional[str] = None
    ) -> Union[dict, None]:

        if order_id:
            order = self._get_order_by_order_id(order_id)
        elif client_order_id:
            order = self._get_order_by_client_order_id(client_order_id)
        else:
            raise ValueError("order_id or client_order_id must be provided!")

        if not order:
            return

        # standardize the order result message if we got one
        if "status" in order.keys():
            return self._standardize_stop_order_response(order)
        else:
            return self._standardize_order_response(order)

    @wrap_call
    def get_fills(
        self,
        order_id=None,
        symbol=None,
        side=None,
        order_type=None,
        start=None,
        end=None,
    ) -> Union[dict, None]:

        got_fills, retry_counter = False, 0

        # sometimes the Kucoin system is not fast enough. this means
        # that it may take more than a few milliseconds before their
        # matching engine updates the database.
        # this may lead to a response which shows that the order went
        # through but some values are missing. we repeat the API call
        # a few times (with increasing delay) if this happens to get
        # the correct result.
        while not got_fills and retry_counter < self.max_retry:
            res = self.trade_client.get_fill_list(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                start=start,
                tradeType="MARGIN_TRADE",
            )
            if res and res["totalNum"] > 0:
                got_fills = True
                return res
            else:
                pprint(res)
                retry_counter += 1
                sleep(retry_counter / 2)

        return

    @wrap_call
    def get_active_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
    ) -> Union[Tuple[dict], None]:

        # we need to make two API calls ...
        # first: active limit orders
        res = self.trade_client.get_order_list(
            symbol=symbol,
            status="active",
            tradeType="MARGIN_TRADE",
        )

        if orders := res["items"] if res else []:
            orders = [self._standardize_order_response(o) for o in orders]
            orders = orders[::-1]

        # second: active stop (limit or market) orders
        stop_orders = self.trade_client.get_all_stop_order_details(
            symbol=symbol, trade_type="MARGIN_TRADE"
        )

        if stop_orders:
            stop_orders = [
                self._standardize_stop_order_response(o) for o in stop_orders["items"]
            ]
            orders += stop_orders[::-1]

        return tuple(orders) if orders else None

    @wrap_call
    def get_active_stop_orders(
        self, symbol: Union[str, None] = None
    ) -> Union[Tuple[dict], None]:
        res = self.trade_client.get_all_stop_order_details()

        if not res:
            return tuple()

        if symbol:
            return tuple(
                self._standardize_stop_order_response(o)
                for o in res["items"]
                if o["symbol"] == symbol
            )

        return tuple(self._standardize_stop_order_response(o) for o in res["items"])

    # .........................................................................
    # helper methods for order retrieval
    @wrap_call
    def _get_multiple_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        order_type: Optional[str] = None,
        status: Optional[str] = "done",
        start: Union[int, str, None] = None,
        end: Union[int, str, None] = None,
        page: Optional[int] = None,
    ) -> Tuple[dict]:

        # use the KucoinOrderDownloader class if start and/or end date
        # was provided in the request
        if any(arg is not None for arg in (start, end)):
            kod = KucoinOrderDownloader(client=self.trade_client)
            orders = kod.get_orders(
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=status,
                start=start,
                end=end,  # type:ignore
            )

        # ... otherwise just download the last seven days or whatever
        # Kucoin considers to be recent orders
        else:
            orders, page = [], 1
            while True:
                print(f"downloading page {page}")
                res = self.trade_client.get_order_list(
                    symbol=symbol,
                    status=status,
                    tradeType="MARGIN_TRADE",
                    page=page,
                    pageSize=500,
                )

                if not res:
                    break

                try:
                    orders += res.get("items")
                except Exception:
                    raise Exception(res.get("msg"))

                if page >= res.get("totalPage"):
                    orders = sorted(orders, key=lambda x: x["createdAt"])
                    break
                else:
                    page += 1

        return tuple(self._standardize_order_response(o) for o in orders)

    @wrap_call
    def _get_order_by_order_id(self, order_id: str) -> Union[dict, None]:
        return self.trade_client.get_order_details(orderId=order_id)

    @wrap_call
    def _get_order_by_client_order_id(self, client_order_id: str) -> Union[dict, None]:

        return self.trade_client.get_client_order_details(clientOid=client_order_id)

    # -------------------------------------------------------------------------
    # NOTE: this section contains some high level functions to make life easier
    # -------------------------------------------------------------------------
    @wrap_call
    def buy_market(
        self,
        symbol: str,
        client_order_id: Optional[str] = None,
        base_qty: Optional[float] = None,
        quote_qty: Optional[float] = None,
        auto_borrow=False,
    ) -> Union[dict, None]:

        if client_order_id is None:
            client_order_id = str(uuid1())

        return self.trade_client.create_market_margin_order(
            symbol=symbol,
            client_oid=client_order_id,
            side="buy",
            size=base_qty,
            funds=quote_qty,
            autoBorrow=auto_borrow,
            marginModel="cross",
        )

    @wrap_call
    def sell_market(
        self,
        symbol: str,
        client_order_id: Optional[str] = None,
        base_qty: Optional[float] = None,
        quote_qty: Optional[float] = None,
        auto_borrow=False,
    ) -> Union[dict, None]:

        if client_order_id is None:
            client_order_id = str(uuid1())

        return self.trade_client.create_market_margin_order(
            symbol=symbol,
            client_oid=client_order_id,
            side="sell",
            size=base_qty,
            funds=quote_qty,
            autoBorrow=auto_borrow,
            marginModel="cross",
        )

    @wrap_call
    def buy_limit(
        self,
        symbol: str,
        price: str,
        base_qty: Optional[str] = None,
        client_order_id: Optional[str] = None,
        margin_mode: str = "cross",
        auto_borrow: bool = False,
        stp: Optional[str] = None,
        remark: Optional[str] = None,
    ) -> Union[dict, None]:

        return self.trade_client.create_limit_margin_order(
            symbol=symbol,
            side="buy",
            size=base_qty,
            price=price,
            client_oid=client_order_id,
            margin_mode=margin_mode,
            auto_borrow=auto_borrow,
            stp=stp,
            remark=remark,
        )

    @wrap_call
    def sell_limit(
        self,
        symbol: str,
        price: str,
        base_qty: Optional[str] = None,
        client_order_id: Optional[str] = None,
        margin_mode: str = "cross",
        auto_borrow: bool = False,
        stp: Optional[str] = None,
        remark: Optional[str] = None,
    ) -> Union[dict, None]:

        return self.trade_client.create_limit_margin_order(
            symbol=symbol,
            side="sell",
            size=base_qty,
            price=price,
            client_oid=client_order_id,
            margin_mode=margin_mode,
            auto_borrow=auto_borrow,
            stp=stp,
            remark=remark,
        )

    # .........................................................................
    @wrap_call
    def stop_limit(
        self,
        symbol: str,
        side: str,
        base_qty: str,
        stop_price: str,
        limit_price: str,
        client_order_id: Optional[str] = None,
        loss_or_entry: str = "loss",
    ) -> Union[dict, None]:

        return self.trade_client.create_limit_stop_order(
            client_oid=client_order_id,
            symbol=symbol,
            side=side.lower(),
            stopPrice=stop_price,
            price=limit_price,
            size=base_qty,
            stop=loss_or_entry,
            tradeType="MARGIN_TRADE",
        )

    @wrap_call
    def stop_market(
        self,
        symbol: str,
        side: str,
        base_qty: str,
        stop_price: str,
        client_order_id: Optional[str] = None,
    ) -> Union[dict, None]:

        loss_or_entry = "loss" if side == "SELL" else "entry"

        return self.trade_client.create_market_stop_order(
            client_oid=client_order_id,
            symbol=symbol,
            side=side.lower(),
            stopPrice=stop_price,
            size=base_qty,
            stop=loss_or_entry,
            tradeType="MARGIN_TRADE",
        )

    # .........................................................................
    def cancel_order(self, order_id: str):
        return self.trade_client.cancel_order(orderId=order_id)

    @wrap_call
    def cancel_all_orders(self, symbol: Optional[str] = None):
        return self.trade_client.cancel_all_orders(
            symbol=symbol, tradeType="MARGIN_TRADE"
        )

    # -------------------------------------------------------------------------
    @wrap_call
    def get_margin_risk_limit(self) -> Union[dict, None]:
        return self.margin_client.get_margin_risk_limit()

    @wrap_call
    def get_borrow_details_for_all(self) -> Union[dict, None]:
        res = self.margin_client.get_margin_account()
        return res["accounts"] if res else res

    @wrap_call
    def get_borrow_details(self, asset: str) -> Union[dict, None]:
        res = self.margin_client.get_margin_account()

        if res:
            accounts = res.get("accounts")
            res = [item for item in accounts if item.get("currency") == asset]

        if res:
            res = res[0]

            # convert all values to float (where possible)
            for k, v in res.items():
                try:
                    res[k] = float(v)
                except Exception:
                    pass

            return res
        else:
            raise ValueError(f"Unknown asset/currency: {asset}")

    @wrap_call
    def get_liability(self, asset: Optional[str]) -> Union[list, None]:
        res = self.margin_client.get_repay_record(currency=asset)
        return items if (items := res.get("items") if res else None) else None

    @wrap_call
    def borrow(
        self,
        currency: str,
        size: float,
        type: str = "FOK",
        max_rate: Optional[float] = None,
        term: Optional[str] = None,
    ):
        return self.margin_client.create_borrow_order(
            currency=currency, size=size, order_type=type, max_rate=max_rate, term=term
        )

    @wrap_call
    def repay(
        self,
        currency: str,
        size: float,
        trade_id: Optional[str] = None,
        sequence: Optional[str] = "HIGHEST_RATE_FIRST",
    ) -> Union[dict, None]:

        # repay a single borrow order - order_id must be
        # provided in that case
        if trade_id:
            res = self.margin_client.repay_single_order(
                currency=currency, size=size, tradeId=trade_id
            )
        # otherwise repay everything in one go
        else:
            res = self.margin_client.click_to_repayment(
                currency=currency, size=size, sequence=sequence
            )

        return res


# =============================================================================
class KucoinCrossMargin(Public, Account, CrossMargin):
    """This is the exchange class for Kucoin that handles all API calls.

    .. note

        All exceptions are catched and all methods return the response in the
        form:

            {'success' : True,
             'message' : result,
             'error' : None,
             'error code' : None,
             'status code' : 200,
             'execution time' : round((time() - _st) * 1000)
             }

        -> for a 'happy' result, and

            {'success' : False,
             'message' : None,
             'error' : e.message,
             'error code' : e.code,
             'status code' : e.status_code,
             'execution time' : round((time() - _st)*1000),
             'arguments' : kwargs
             }

        -> for an 'unhappy' result (= an exception occured)

    :param market: 'spot' or 'margin' (only margin implemented for now)
    :type market: str
    """

    def __init__(self, credentials: Union[dict, None] = None):

        self.name = "Kucoin Digital Asset Exchange"
        self.market = "cross margin"
        self.credentials = credentials or CREDENTIALS

        Public.__init__(self)
        Account.__init__(self, credentials=self.credentials)
        CrossMargin.__init__(self, credentials=self.credentials)

        self.max_workers = 25
        self.max_retry = 5
        self.delay = 0
        self.limit = 1499

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False


class KucoinFactory:

    def build_client(self, market: str, credentials: dict):
        if market.lower() == "cross margin":
            return KucoinCrossMargin(credentials=credentials)

        elif market.lower() == "spot":
            raise NotImplementedError("SPOT: still waiting to be implemented!")

        else:
            raise ValueError(f"{market} is not a valid market")
