#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 23:54:28 2022

@author: dhaneor


    "rateLimits": [{
                        "rateLimitType": "REQUEST_WEIGHT",
                        "interval": "MINUTE",
                        "intervalNum": 1,
                        "limit": 1200
                    }, {
                        "rateLimitType": "ORDERS",
                        "interval": "SECOND",
                        "intervalNum": 10,
                        "limit": 50
                    }, {
                        "rateLimitType": "ORDERS",
                        "interval": "DAY",
                        "intervalNum": 1,
                        "limit": 160000
                    }, {
                        "rateLimitType": "RAW_REQUESTS",
                        "interval": "MINUTE",
                        "intervalNum": 5,
                        "limit": 6100
                    }]
"""

import os
import sys
import concurrent.futures
import pandas as pd
import logging

from uuid import uuid1
from time import time, sleep
from typing import Union
from configparser import ConfigParser
from pprint import pprint
from threading import Lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# adding the parent directory to the search path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

# from mock_responses.binance_client import Client
from binance.client import Client  # noqa: E402
from binance.exceptions import (  # noqa: E402
    BinanceAPIException,
    BinanceRequestException,
)
from .util.ohlcv_download_prepper import OhlcvDownloadPrepper  # noqa: E402
from util.timeops import (  # noqa: E402
    unix_to_utc,
    utc_to_unix,
    get_start_and_end_timestamp,
    seconds_to_next_full,
)

# =============================================================================
JUST_STARTED = True
CLIENT = None
GO_AHEAD = True
MAX_WORKERS = 2  # don't change here will be set by the instance!
WEIGHTS = (0, 0)
CALLBACK = None
VERBOSE = False


# =============================================================================
class call_wrapper:
    def __init__(self, debug=False):
        self.debug = debug
        self.client = CLIENT

        self.delay = 0
        self.too_many_requests = False
        self.banned = False
        self.banned_until = 0

    def get_weights(self) -> tuple:
        if self.debug:
            try:
                short = self.client.get("x-mbx-used-weight-1m", 0)
                long = self.client.get("x-mbx-used-weight")
            except Exception:
                return 10000, 10000
        else:
            with Lock():
                short = int(
                    self.client.response.headers.get("x-mbx-used-weight-1m", 500)
                )
                long = int(self.client.response.headers.get("x-mbx-used-weight", 1000))

        return short, long

    def chill_out_bro(self) -> None:
        with Lock():
            weight_used_1m, weight_used = WEIGHTS
        weight_used += MAX_WORKERS
        weight_used_1m += MAX_WORKERS
        out = f">> request weight used: \t{weight_used_1m} {weight_used} "
        out += f"(workers: {MAX_WORKERS})"

        # calculate the delay we need to prevent 'too many requests'
        limit, hard_limit, barrier_limit = 700, 1000, 1200
        d_short, d_long = 0, 0

        if weight_used_1m >= limit:
            if weight_used_1m >= hard_limit:
                more_delay = ((weight_used_1m - hard_limit) ** 3) / 400_000
            else:
                more_delay = 0

            d_short = round((weight_used_1m - limit) / 50 + more_delay, 2)

            if weight_used_1m >= barrier_limit:
                seconds_left = seconds_to_next_full("minute") + 0.5
                d_short = seconds_left
                max_workers = max(1, int(60 / (60 - seconds_left)))
                if max_workers < MAX_WORKERS:
                    CALLBACK(max_workers)

        limit *= 3
        if weight_used > limit:
            if weight_used > 5000:
                more_delay = 5
            d_long = (weight_used - limit) / 50

        delay = d_short + d_long
        out += f"-> {delay:.2f}s delay"
        logger.debug(out)
        sleep(delay)

    # ..........................................................................
    def __call__(self, func):
        def wrapped(*args, **kwargs):
            sleep(0.05)
            _st = time()
            self.client = CLIENT

            if not CLIENT:
                return {
                    "success": False,
                    "message": None,
                    "error": "No client for connection",
                    "error code": "13",
                    "exception": "ConnectionError",
                    "status code": "404",
                    "execution time": round((time() - _st) * 1000),
                    "arguments": kwargs,
                }

            self.chill_out_bro()

            try:
                result = func(*args, **kwargs)

                with Lock():
                    global WEIGHTS
                    WEIGHTS = self.get_weights()

                return {
                    "success": True,
                    "message": result,
                    "status code": 200,
                    "warning": None,
                    "execution time": round(((time() - _st) * 1000)),
                }

            except (BinanceAPIException, BinanceRequestException) as e:
                return {
                    "success": False,
                    "message": None,
                    "error": e.message,
                    "error code": e.code,
                    "exception": type(e).__name__,
                    "status code": e.status_code,
                    "execution time": round((time() - _st) * 1000),
                    "arguments": kwargs,
                }

            except ConnectionError as e:
                return {
                    "success": False,
                    "message": None,
                    "error": e,
                    "error code": "13",
                    "exception": "ConnectionError",
                    "status code": "404",
                    "execution time": round((time() - _st) * 1000),
                    "arguments": kwargs,
                }

            except AttributeError as e:
                error_str = str(e)
                print(error_str)
                if "'NoneType'" in error_str:
                    error_str = "No connection to host"
                    error_type = "ConnectionError"
                else:
                    error_type = type(e).__name__

                return {
                    "success": False,
                    "message": None,
                    "error": error_str,
                    "error code": "999",
                    "exception": error_type,
                    "status code": "000",
                    "execution time": round((time() - _st) * 1000),
                    "arguments": kwargs,
                }

            except Exception as e:
                return {
                    "success": False,
                    "message": None,
                    "error": e,
                    "error code": "999",
                    "exception": type(e).__name__,
                    "status code": "000",
                    "execution time": round((time() - _st) * 1000),
                    "arguments": kwargs,
                }

        return wrapped


# =============================================================================
class BinancePublic:
    """This class handles all the public calls to the Binance API."""

    INTERVALS = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
    }

    def __init__(self, max_workers=10):

        self.name = "Binance Digital Asset Exchange"
        self.api_key = None
        self.api_secret = None
        self.client = None

        self._max_workers_restricted = False
        self.set_max_workers(max_workers=max_workers)

        self.go_ahead = True
        self._stop_it = False
        self.max_retry = 5
        self._request_delay = 0

        self._import_api_key()
        self._connect()

        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    # -------------------------------------------------------------------------
    def _connect(self):
        try:
            self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
            global CLIENT
            CLIENT = self.client
        except Exception as e:
            # print(e)
            self._extract_ban_time(e)

    def _import_api_key(self):

        c = ConfigParser()
        c.read("config.ini")

        self.api_key = ""  # c["API KEY BINANCE"]["api_key"]
        self.api_secret = ""  # c["API KEY BINANCE"]["api_secret"]

        return

    def _extract_ban_time(self, error):
        e = str(error)
        if "IP banned" in e:
            e = e.split(":")[1]
            e = e.split(";")[1]
            e = e.split(".")[0]

            ts = e.split(" ")[4]
            banned_until = unix_to_utc(int(ts))

            print(f"{e} - banned until {banned_until}")

        else:
            return int(time()) + 100_000

    # -------------------------------------------------------------------------
    # methods to get general information (time, status, markets ...) from
    # the exchange
    @call_wrapper(debug=False)
    def get_server_time(self):
        res = self.client.get_server_time()
        try:
            res = res["serverTime"]
        except Exception:
            pass
        return res

    @call_wrapper(debug=False)
    def get_server_status(self):
        """Get the current system status

        :return: returns the system status
        :rtype: dict

        .. code:: python
            {
                "status": 0,      // 0 / 1
                "msg": "normal"   // "normal" / "system_maintenance"
            }
        """
        res = self.client.get_system_status()

        # convert status code to string (for compatibility with Kucoin format)
        try:
            res["status"] = "open" if res["status"] == 0 else "close"
        except Exception:
            pass

        return res

    @call_wrapper(debug=False)
    def get_currencies(self):
        """Get all coins/currencies that are on the exchange.

        NOTE: The formats for Binance and Kucoin are very different, so
        before using the results they must be converted. Didn't do this
        here as it doesn't make sense to me and they contain pretty much
        different informations.

        I decided that it is easier (and gives additional information) to
        retrieve coinsinfo from coingecko and use these where necessary.

        :return: all the coins and associated information
        :rtype: list

        .. code:: python

            [{'coin': 'XMR',
            'depositAllEnable': True,
            'free': '0.001313',
            'freeze': '0',
            'ipoable': '0',
            'ipoing': '0',
            'isLegalMoney': False,
            'locked': '0',
            'name': 'Monero',
            'networkList': [{'addressRegex': '^[48][a-zA-Z|\\d]{94}([a-zA-Z|\\d]{11})?$',  # noqa E501
                            'addressRule': '',
                            'coin': 'XMR',
                            'depositDesc': '',
                            'depositEnable': True,
                            'isDefault': True,
                            'memoRegex': '',
                            'minConfirm': 3,
                            'name': 'Monero',
                            'network': 'XMR',
                            'resetAddressStatus': False,
                            'sameAddress': False,
                            'specialTips': '',
                            'unLockConfirm': 0,
                            'withdrawDesc': '',
                            'withdrawEnable': True,
                            'withdrawFee': '0.0001',
                            'withdrawIntegerMultiple': '0.00000001',
                            'withdrawMax': '10000000000',
                            'withdrawMin': '0.0002'}],
            'storage': '0',
            'trading': True,
            'withdrawAllEnable': True,
            'withdrawing': '0'
            }]
        """
        return self.client.get_all_coins_info()

    # TODO  we need a module that queries the coingecko API and then we can
    #       build a markets section that is similar between Binance and
    #       Kucoin (and possible future integrations). Kucoin returns its
    #       own classification, which is kind of arbitrary and Binance
    #       doesn't have something like this at all.
    @call_wrapper(debug=False)
    def get_markets(self):
        all_symbols = self.client.get_exchange_info()
        return set([s["quoteAsset"] for s in all_symbols["symbols"]])

    def get_symbols(self, quote_asset: str = None):
        self.market = "spot"

        res = self._get_exchange_info()

        if res["success"]:
            symbols = res["message"]["symbols"]

            if "margin" in self.market:
                symbols = [s for s in symbols if "MARGIN" in s["permissions"]]

            if quote_asset:
                symbols = [s for s in symbols if s["quoteAsset"] == quote_asset]

            res["message"] = symbols

        return res

    def get_symbol(self, symbol: str = None) -> dict:
        if symbol:
            res = self.get_symbols()

            if not res["success"]:
                return res

            all_symbols = res["message"]
            for _symbol in all_symbols:
                if _symbol["symbol"] == symbol.upper():
                    res["message"] = _symbol
                    return res

        return {
            "success": False,
            "message": None,
            "error": "symbol not exists",
            "error code": 900001,
            "status code": 200,
            "execution time": 0,
            "arguments": symbol,
        }

    @call_wrapper(debug=False)
    def get_ticker(self, symbol: str = None):
        return self._standardize_ticker(self.client.get_ticker(symbol=symbol))

    # -------------------------------------------------------------------------
    # method to get historical ohlcv data
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Union[int, str] = None,
        end: Union[int, str] = None,
        as_dataframe: bool = True,
    ) -> list:

        if not self.client:
            return {
                "success": False,
                "message": None,
                "error": "No client",
                "error code": 429,
                "status code": 429,
                "execution time": 0,
                "arguments": None,
            }

        dl_request = self._prepare_request(
            symbol=symbol, start=start, end=end, interval=interval
        )

        _results = []
        if self.verbose:
            print(dl_request.get("number of chunks"), len(dl_request.get("chunks")))

        # ......................................................................
        # download the data in parallel with threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:

            futures = []
            for chunk in dl_request["chunks"]:

                kwargs = {
                    "symbol": dl_request.get("symbol"),
                    "interval": dl_request.get("interval"),
                    "start_ts": chunk[0],
                    "end_ts": chunk[1],
                }

                # if self.verbose : pprint(kwargs)

                futures.append(executor.submit(self._fetch_ohlcv, **kwargs))

            futures, _ = concurrent.futures.wait(futures)
            for future in futures:
                _results.append(future.result())

        # extract data, errors and status codes from results
        res = [
            _res.get("message") for _res in _results if _res.get("message") is not None
        ]

        warnings = [
            _res["warning"] for _res in _results if _res.get("warning") is not None
        ]

        errors = [_res["error"] for _res in _results if _res.get("error") is not None]

        error_codes = [
            _res["error code"]
            for _res in _results
            if _res.get("error code") is not None
        ]

        status_codes = [_res.get("status code") for _res in _results]

        execution_times = [_res.get("execution time") for _res in _results]

        if self.verbose:
            if warnings:
                pprint(warnings)
            if errors:
                pprint(errors)

        # we now have a list of lists and need to flatten it
        res = [item for sub_list in res for item in sub_list]

        # convert the result to  a dataframe if parameter as_dataframe is True
        if as_dataframe:
            res = self._klines_to_dataframe(res)

        return {
            "success": True,
            "message": res,
            "error": errors,
            "error code": error_codes,
            "status code": status_codes[-1],
            "execution time": round(sum(execution_times) / len(_results)),
            "arguments": None,
        }

    def get_ohlcv_old(
        self,
        symbol: str,
        interval: str,
        start: Union[int, str] = None,
        end: Union[int, str] = None,
        as_dataframe: bool = True,
    ) -> dict:

        # check format of start and end time and convert to seconds,
        # if necessary
        if not isinstance(start, int) or not isinstance(end, int):
            start_ts, end_ts = get_start_and_end_timestamp(
                start, end, interval, unit="seconds", verbose=False
            )
        else:
            start_ts, end_ts = start, end

        # _hs = unix_to_utc(start_ts)
        # _he = unix_to_utc(end_ts)
        # print(_hs, _he)

        # get the actual data from our wrapped function below
        res = self._fetch_ohlcv(
            symbol=symbol, interval=interval, start=start_ts, end=end_ts
        )

        if res["success"] and as_dataframe:
            res["message"] = self._klines_to_dataframe(res["message"])

        if res["status code"] == 429:
            res["warning"] = "Hit request rate limit!"

        return res

    @call_wrapper(debug=False)
    def _fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ts: int = None,
        end_ts: int = None,
        limit: int = 1000,
    ) -> dict:
        """This method fetches the klines from the API. It should not be
        called directly because almost always we need to some checks before
        calling the API. All of this is implemented in  get_ohlcv() and
        this is the method that the caller should use.
        """
        return self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=str(start_ts),
            end_str=str(end_ts),
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # helper methods
    @call_wrapper(debug=False)
    def _get_exchange_info(self):
        return self.client.get_exchange_info()

    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """This method converts the list that we got from the
        Binance API to a dataframe"""
        # set column names
        columns = [
            "open time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close time",
            "quote volume",
            "number of trades",
            "taker base volume",
            "taker quote volume",
            "ignore",
        ]

        # build dataframe from list of klines
        df = pd.DataFrame(columns=columns, data=klines)

        # drop superfluous columns
        df.drop(columns=["ignore"], axis=1, inplace=True)
        df.drop(columns=["taker base volume"], axis=1, inplace=True)
        df.drop(columns=["taker quote volume"], axis=1, inplace=True)

        # convert values to numeric
        float_columns = ["open", "high", "low", "close", "volume", "quote volume"]
        int_columns = ["open time", "close time", "number of trades"]

        for col in df.columns:
            df[col] = df[col].astype(float) if col in float_columns else df[col]
            df[col] = df[col].astype(int) if col in int_columns else df[col]

        # add column with human readable 'open time'
        df.insert(1, "human open time", pd.to_datetime(df["open time"], unit="ms"))

        df.sort_values(by="open time", inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df

    @call_wrapper(debug=False)
    def _get_earliest_valid_timestamp(self, symbol: str, interval: str = "1d") -> int:
        try:
            ts = self.client._get_earliest_valid_timestamp(
                symbol=symbol, interval=interval
            )
        except Exception:
            logger.warning(
                f"Could not get earliest valid timestamp "
                f"for {symbol} ({interval}) on Binance"
            )
            ts = int(utc_to_unix("2017-07-01 00:00:00") / 1000)
        finally:
            return ts

    def set_max_workers(self, max_workers: int, no_further_updates: bool = False):
        if not self._max_workers_restricted:
            self._max_workers = max_workers
            global MAX_WORKERS
            MAX_WORKERS = max_workers

        if no_further_updates:
            self._max_workers_restricted = True

    # see client.py for additional methods that we don't need for now ...

    # ..........................................................................
    def _test_wrapper(self):
        global CLIENT, JUST_STARTED
        # print('just started: ', JUST_STARTED)

        if JUST_STARTED:
            CLIENT = {"x-mbx-used-weight-1m": 0, "x-mbx-used-weight": 0}
            pprint(CLIENT)
            JUST_STARTED = False
        return self._test_wrapper_helper(x=1, y=2)

    @call_wrapper(debug=True)
    def _test_wrapper_helper(self, **kwargs):
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        # print('inside wrapped method')

        CLIENT["x-mbx-used-weight"] += 1
        if int(CLIENT["x-mbx-used-weight-1m"]) > 1050:
            CLIENT["x-mbx-used-weight-1m"] = 0
        else:
            CLIENT["x-mbx-used-weight-1m"] += 1
        # pprint(CLIENT)
        return [x, y]


# =============================================================================
class BinanceSpot:
    """This class handles all the private (=require 'Trade' permission)
    calls to the Binance API for 'Cross Margin' trading."""

    def __init__(self):
        self.account = "spot"

    # -------------------------------------------------------------------------
    # SPOT ACCOUNT related methods
    @call_wrapper(debug=False)
    def get_account(self) -> dict:
        res = self.client.get_account()

        try:
            accounts = res["balances"]
        except Exception:
            return res

        res = [self._standardize_account(acc) for acc in accounts]
        return res

    @call_wrapper(debug=False)
    def get_balance(self, asset: str = None) -> dict:
        res = self.client.get_account()

        try:
            accounts = res["balances"]
            balance = [acc for acc in accounts if acc["asset"] == asset][0]
        except Exception:
            return res

        return self._standardize_account(balance)

    @call_wrapper(debug=False)
    def get_fees(self, symbol: str = None) -> dict:
        res = self.client.get_trade_fee(symbol=symbol)
        if res:
            return self._standardize_fees(res[0])
        else:
            return res

    # -------------------------------------------------------------------------
    # QUERYING, CREATING and DELETING ORDERS related
    #
    def get_orders(
        self,
        symbol: str,
        side: str = None,
        order_type=None,
        status=None,
        start: Union[int, str] = None,
        end: Union[int, str] = None,
        limit: int = 1000,
    ) -> dict:

        if start or end:
            # if start and/or end parameter was given, get the timestamps
            start_ts, end_ts = get_start_and_end_timestamp(
                start, end, "1d", unit="milliseconds", verbose=True
            )

            # TODO  We are only allowed to get orders for a period of
            #       24 hours if we use start/end. So, here should be an
            #       implementation of a loop that sends multiple requests
            #       if we have a longer period. In most cases it makes
            #       more sense to omit these parameters and instead get
            #       (up to) the last 1000 orders which can be done in one
            #       call (see else branch below)
            end_ts = start_ts + 24 * 60 * 60000 - 1

            # query orders for given period
            res = self._get_all_orders(symbol=symbol, start=start_ts, end=end_ts)
        else:
            # ... otherwise query orders without constraining period
            res = self._get_all_orders(symbol=symbol, limit=limit)

        # .....................................................................
        # filter orders for given criteria (status, side, ...)
        if res["success"]:
            orders = res["message"]
            # filter for order side if necessary
            if side and side in ["BUY", "SELL"]:
                orders = [o for o in orders if o["side"] == side]
            # filter for order type if necessary
            valid_order_types = [
                "MARKET",
                "LIMIT",
                "STOP_LOSS",
                "STOP_LOSS_LIMIT",
                "TAKE_PROFIT",
                "TAKE_PROFIT_LIMIT",
                "LIMIT_MAKER",
            ]

            if order_type and order_type in valid_order_types:
                orders = [o for o in orders if o["type"] == order_type]
            # filter for status if necessary
            if status:
                orders = [o for o in orders if o["status"] == status]
            # filter for start time if necessary
            if start:
                orders = [o for o in orders if o["time"] >= start_ts]
            # filter for end time if necessary
            if end:
                orders = [o for o in orders if o["time"] <= end_ts]
            # replace original response with filtered results
            res["message"] = orders
        # .....................................................................
        return res

    @call_wrapper(debug=False)
    def get_order(
        self, symbol: str, order_id: str = None, client_order_id: str = None
    ) -> dict:

        # get the result for a certain 'order id'
        res = self.client.get_order(
            symbol=symbol, orderId=order_id, origClientOrderId=client_order_id
        )

        return res

    @call_wrapper(debug=False)
    def get_active_orders(
        self, symbol: str = None, side: str = None, order_type=None
    ) -> dict:

        # to make this method behave the same as the Kucoin version,
        # if nor order type is given, then only LIMIT orders will
        # be returned. use get_active_stop_orders to retrieve all
        # stop orders!
        if order_type is None:
            order_type = "LIMIT"

        orders = self.client.get_open_orders(symbol=symbol)

        # filter for side and order type if those parameters are not None
        if orders:
            if side:
                orders = [o for o in orders if o["side"] == side.upper()]
            if order_type:
                order_type = "_".join([item.upper() for item in order_type.split("_")])
                orders = [o for o in orders if o["type"] == order_type]

        return orders

    @call_wrapper(debug=False)
    def get_active_stop_orders(self, symbol: str = None, side: str = None) -> dict:

        orders = self.client.get_open_orders(symbol=symbol)

        stop_types = [
            "STOP_LOSS_LIMIT",
            "STOP_LOSS_MARKET",
            "TAKE_PROFIT_LIMIT",
            "TAKE_PROFIT_MARKET",
        ]

        # filter for all stop type orders
        orders = [o for o in orders if o["type"] in stop_types]

        if side:
            orders = [o for o in orders if o["side"] == side.upper()]

        return orders

    # .........................................................................
    @call_wrapper(debug=False)
    def _get_all_orders(
        self, symbol: str = None, start: int = None, end: int = None, limit=1000
    ) -> dict:

        return self.client.get_all_orders(
            symbol=symbol, limit=limit, startTime=start, endTime=end
        )

    # -------------------------------------------------------------------------
    # NOTE: this section contains some high level functions to make life easier
    # -------------------------------------------------------------------------
    def sell_all(self, symbol: str) -> dict:

        _st = time()

        def _round_qty(qty: float):
            base_qty = round(qty, filters["base precision"])
            if base_qty > qty:
                base_qty -= filters["base step"]
            return base_qty

        # .....................................................................

        self.cancel_all_orders(symbol)
        filters = self._get_symbol_filters(symbol)
        if not filters:
            return {
                "success": False,
                "message": None,
                "error": "Unable to get filters from exchange",
                "error code": "11",
                "exception": "General Exception",
                "status code": 400,
                "execution time": round((time() - _st) * 1000),
                "arguments": f"symbol {symbol}",
            }

        base_asset = filters["base asset"]

        res = self.get_balance(base_asset)

        if res["success"]:
            balance = res["message"]
            qty = float(balance["free"])
            base_qty = _round_qty(qty)

            if base_qty > 0:
                return self.sell_market(
                    symbol=symbol, base_qty=base_qty, client_order_id=str(uuid1())
                )
            else:
                return {
                    "success": False,
                    "message": None,
                    "error": f"Low balance ({qty} {base_asset})",
                    "error code": 23,
                    "status code": 0,
                    "execution time": round((time() - _st) * 1000),
                    "arguments": f"{symbol=}",
                }
        else:
            return res

    # .........................................................................
    @call_wrapper(debug=False)
    def buy_market(
        self,
        symbol: str,
        client_order_id: str,
        base_qty: float = None,
        quote_qty: float = None,
        auto_borrow=False,
        base_or_quote="quote",
    ) -> dict:

        if base_qty and quote_qty:
            if base_or_quote == "quote":
                base_qty = None
            else:
                quote_qty = None

        return self.client.order_market_buy(
            symbol=symbol,
            newClientOrderId=client_order_id,
            quantity=base_qty,
            quoteOrderQty=quote_qty,
            newOrderRespType="FULL",
        )

    @call_wrapper(debug=False)
    def sell_market(
        self,
        symbol: str,
        client_order_id: str,
        base_qty: float = None,
        quote_qty: float = None,
    ) -> dict:

        return self.client.order_market_sell(
            symbol=symbol,
            newClientOrderId=client_order_id,
            quantity=base_qty,
            quoteOrderQty=quote_qty,
            newOrderRespType="FULL",
        )

    @call_wrapper(debug=False)
    def buy_limit(
        self,
        symbol=None,
        base_qty=None,
        price=None,
        client_oid=None,
        margin_mode=None,
        auto_borrow=False,
        stp=None,
        remark=None,
    ) -> dict:

        return self.client.create_order(
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            quantity=base_qty,
            price=price,
            newClientOrderId=str(client_oid),
            newOrderRespType="FULL",
            timeInForce="GTC",
        )

    # .........................................................................
    @call_wrapper(debug=False)
    def stop_limit(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        stop_price: float,
        limit_price: float,
        client_order_id: str,
        loss_or_entry: str = "loss",
        mode: int = 0,
    ) -> dict:

        return self.client.create_order(
            symbol=symbol,
            side=side.upper(),
            type="STOP_LOSS_LIMIT",
            stopPrice=stop_price,
            price=limit_price,
            quantity=base_qty,
            newOrderRespType="FULL",
            newClientOrderId=client_order_id,
            timeInForce="GTC",
        )

    # .........................................................................
    def cancel_order(self, symbol: str, order_id: str, stop_order: bool = False):
        return self._cancel_order_by_order_id(symbol=symbol, order_id=order_id)

    def cancel_all_orders(self, symbol: str = None):

        res = self._get_all_orders(symbol=symbol)

        if res["success"]:
            orders = [o for o in res["message"] if o["status"] == "NEW"]
            order_ids = [o["orderId"] for o in orders]

            res["message"] = [
                self._cancel_order_by_order_id(symbol=symbol, order_id=oid)
                for oid in order_ids
            ]
            return res

    # .........................................................................
    # NOTE: these internal methods should not be called directly! instead
    # use the ones above which are more specific and easier to use
    # .........................................................................
    #
    # CREATE order(s)
    @call_wrapper(debug=False)
    def _create_order(
        self,
        symbol=None,
        side=None,
        remark=None,
        order_type=None,
        client_oid=None,
        stp=None,
        margin_mode=None,
        auto_borrow=False,
        size=None,
        price=None,
        funds=None,
    ):

        if client_oid is None:
            client_oid = str(uuid1())

        return self.client.create_margin_order(
            symbol=symbol,
            side=side.lower(),
            order_type=order_type,
            remark=remark,
            client_oid=client_oid,
            stp=stp,
            margin_mode=margin_mode,
            auto_borrow=auto_borrow,
            size=size,
            price=price,
            funds=funds,
        )

    @call_wrapper(debug=False)
    def _create_stop_limit_order(
        self,
        symbol: str = None,
        side: str = None,
        base_qty: float = None,
        client_oid: str = None,
        stop_price: float = None,
        limit_price: float = None,
        loss_or_entry: str = "loss",
        remark: str = None,
    ) -> dict:

        return self.client.create_limit_order(
            symbol=symbol,
            side=side,
            stop_price=stop_price,
            price=limit_price,
            size=base_qty,
            client_oid=client_oid,
            remark=remark,
            stop=loss_or_entry,
            trade_type="MARGIN_TRADE",
        )

    @call_wrapper(debug=False)
    def _create_stop_market_order(self):
        pass

    # .........................................................................
    # CANCEL order(s)
    @call_wrapper(debug=False)
    def _cancel_order_by_order_id(self, symbol: str, order_id: str):
        return self.client.cancel_order(symbol=symbol, orderId=int(order_id))


# =============================================================================
class Binance(BinancePublic, BinanceSpot, OhlcvDownloadPrepper):
    """This is the exchange class for Binance that handles all API calls.

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

    def __init__(self, market: str = "spot", verbose: bool = True):

        # TODO  implement that 'market' is read from config.ini if no
        #       market was provided
        if not market:
            raise Exception("Please provide a value for market!")

        BinancePublic.__init__(self)
        OhlcvDownloadPrepper.__init__(self)

        if market.lower() == "spot":
            BinanceSpot.__init__(self)
        elif market.lower() == "cross margin":
            raise NotImplementedError
        elif market.lower() == "isolated margin":
            raise NotImplementedError
        else:
            print(f"unknown market: {market}")
            sys.exit()

        self.market = market
        self._max_simultaneous_requests = 2  # ohlcv download threads
        self.limit = 1000  # max number of klines for ohlcv download
        self.verbose = False  # verbose # silent or talkative operation

        global CALLBACK
        CALLBACK = self.set_max_workers

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    # -------------------------------------------------------------------------
    # methods to standardize the response from the Binance API
    def _standardize_ticker(self, ticker: dict) -> dict:
        """Convert ticker response to standard format

        :param ticker: Binance API response for a single ticker
        :type ticker: dict

        .. code:: python
            {
                'askPrice': '0.00002037',
                'askQty': '28985.00000000',
                'bidPrice': '0.00002035',
                'bidQty': '39976.00000000',
                'closeTime': 1645304955649,
                'count': 72184,
                'firstId': 135080471,
                'highPrice': '0.00002124',
                'lastId': 135152654,
                'lastPrice': '0.00002035',
                'lastQty': '41.00000000',
                'lowPrice': '0.00001925',
                'openPrice': '0.00001926',
                'openTime': 1645218555649,
                'prevClosePrice': '0.00001927',
                'priceChange': '0.00000109',
                'priceChangePercent': '5.659',
                'quoteVolume': '1000.15858303',
                'symbol': 'XRPBTC',
                'volume': '49336366.00000000',
                'weightedAvgPrice': '0.00002027'
            }

        :return: Standardized ticker response
        :rtype: dict
        """

        change_pct = str(round(float(ticker["priceChangePercent"]) / 100, 4))

        return {
            "symbol": ticker["symbol"],
            "ask": ticker["askPrice"],
            "bid": ticker["bidPrice"],
            "last": ticker["lastPrice"],
            "open": ticker["openPrice"],
            "high": ticker["highPrice"],
            "low": ticker["lowPrice"],
            "priceChange": ticker["priceChange"],
            "priceChangePercent": change_pct,
            "volume": ticker["volume"],
            "quoteVolume": ticker["quoteVolume"],
        }

    def _standardize_account(self, item: dict) -> dict:
        """Transform one item (asset) to standardized format.

        NOTE:   This only works for SPOT account. As I'm legally not allowed
                to use Binance Margin - I can't test anything related to
                margin trading, because Binance will block all requests.

                So, I implemented the method for items from a margin account
                on a theoretical basis!!!

        :param item: dictionary with the values for one asset in user account
        :type item: dict

        When querying the account, we get a list of dictionaries from
        Binance. Each dictionary represents one asset (currency) and has
        the format:

        .. code:: python

            {
            'asset': 'WOO',
            'free': '0.00000000',
            'locked': '0.00000000'
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
        decimals = str((item["free"].split("."))[1])
        precision = len(decimals)
        free = float(item["free"])
        locked = float(item["locked"])
        total = round(free + locked, precision)

        is_margin = True if "borrowed" in item.keys() else False

        # spot account/balance
        if not is_margin:
            return {
                "asset": item["asset"],
                "free": item["free"],
                "locked": item["locked"],
                "borrowed": "0." + precision * "0",
                "total": total,
            }

        # margin account/balance
        else:
            return {
                "asset": item["asset"],
                "free": item["free"],
                "locked": item["locked"],
                "borrowed": item["borrowed"],
                "interest": item["interest"],
                "total": total,
            }

    def _standardize_fees(self, fees: dict):
        """Convert fee information to standard format

        :param fees: dictionary with fee information for one symbol
        :type fees: dict

        .. code:: python
            {
                'makerCommission': '0.001',
                'symbol': 'XRPUSDT',
                'takerCommission': '0.001'
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
            "maker": fees["makerCommission"],
            "taker": fees["takerCommission"],
        }

    # -------------------------------------------------------------------------
    def _get_symbol_filters(self, symbol: str) -> dict:
        """Get the symbol info (filters etc.)

        :param symbol: name of the symbol
        :type symbol: str

        :returns: a dictionary with filter informations for the symbol
        :rtype: dict
        """
        res = self.get_symbol(symbol)

        if res["success"]:
            base_asset = res["message"]["baseAsset"]
            quote_asset = res["message"]["quoteAsset"]
            filters = res["message"]["filters"]

            try:
                _filter = [f for f in filters if f["filterType"] == "PRICE_FILTER"][0]
                price_step = str(float(_filter["tickSize"]))
                price_precision = len(price_step.split(".")[1])
                print(f"{price_step=} :: {price_precision}")
            except Exception as e:
                pprint(_filter)
                print("-" * 80)
                print(e)
                return

            try:
                _filter = [f for f in filters if f["filterType"] == "LOT_SIZE"][0]
                base_step = str(float(_filter["stepSize"]))
                base_precision = len(base_step.split(".")[1])
                base_step = float(base_step)
                print(f"{base_step=} :: {base_precision}")

            except Exception as e:
                pprint(_filter)
                print("-" * 80)
                print(e)
                return
        else:
            pprint(res)
            return

        return {
            "base asset": base_asset,
            "quote asset": quote_asset,
            "price step": price_step,
            "price precision": price_precision,
            "base step": base_step,
            "base precision": base_precision,
        }
