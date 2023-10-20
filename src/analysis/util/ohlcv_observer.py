#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:57:06 2021

@author: dhaneor
"""
import time
import datetime
import logging
import concurrent.futures
import pandas as pd

from threading import Thread
from typing import Union, Callable, List,Tuple, Iterable
from collections import namedtuple, deque
from statistics import mean
from random import random

from exchange.exchange import ExchangePublic
from .ohlcv_validator import OhlcvValidator
from exchange.util.cryptowatch import CryptoWatch

from broker.config import CREDENTIALS

logger = logging.getLogger('main.ohlcv_observer')
logger.setLevel(logging.DEBUG)

VALID_INTERVALS = [
    '1m', '5m', '15m', '30m','1h', '2h', '4h', '8h', '12h', '1d', '1W', '1M'
]

NUMBER_OF_INTERVALS = 300

# =============================================================================

"""
TODO    ...
"""

# =============================================================================
class TimeDifferenceWatcher(Thread):
    """Class that keeps track of the time difference to the exchange server."""

    def __init__(self, callback:Callable):

        super().__init__()
        self.exchange = ExchangePublic(exchange='kucoin')
        self.callback = callback
        self.time_differences = deque([])
        self._initialize_deque()

    def run(self):

        while True:

            try:
                query_result = self.exchange.get_server_time()
                self._handle_query_result(query_result)
            except Exception as e:
                logger.warning(f'failed to get server time: {e}')

            self.callback(int(mean(self.time_differences)))

            sleep_time = round(60 + (random() * 30), 2)
            time.sleep(sleep_time)

    def _initialize_deque(self):
        for _ in range(2):
            try:
                self._handle_query_result(
                    self.exchange.get_server_time()
                )
            except:
                self.time_differences.append(150)

    def _handle_query_result(self, query_result:dict):
        server_time = self._extract_server_time_from_query_result(query_result)

        if server_time is None:
            return

        latency = self._extract_latency_from_query_result(query_result)
        server_time = server_time + latency

        time_difference = self._get_time_difference_in_ms_to_local_time(
            server_time=server_time
        )

        self.time_differences.append(time_difference)
        if len(self.time_differences) > 10:
            self.time_differences.popleft()

    def _extract_server_time_from_query_result(self, query_result:dict
                                               ) -> Union[int, None]:
        if query_result.get('success'):
            return query_result.get('message', 0)

    def _extract_latency_from_query_result(self, query_result:dict) -> int:
        roundtrip = query_result.get('execution time', 0)
        if roundtrip:
            return int(roundtrip / 2)
        else:
            return 0

    def _get_time_difference_in_ms_to_local_time(self, server_time:int) -> float:
        dt = datetime.datetime.now(datetime.timezone.utc)
        utc_time = dt.replace(tzinfo=datetime.timezone.utc)
        local_time = utc_time.timestamp() * 1000

        return server_time - local_time

class Chronos(Thread):
    """Class that monitors time and notifies the client about
    closed intervals"""

    def __init__(self, callback:Callable):

        Thread.__init__(self)

        self.next_minute: int = -1
        self.next_hour: int = -1
        self.sleep_time: float = 0.1
        self.expired_intervals : list = []

        self.time_difference_to_server: float = 0
        self.time_difference_watcher = TimeDifferenceWatcher(
            self.set_time_difference_to_server
        )
        self.time_difference_watcher.start()

        self.callback = callback

        logger.debug('Chronos is running ...')

    # -------------------------------------------------------------------------
    def run(self):

        now = datetime.datetime.utcnow()

        if self.next_minute == -1:
            delta = datetime.timedelta(minutes=1)
            self.next_minute = (now + delta).minute

        if self.next_hour == -1:
            self.next_hour = now.hour + 1

        while True:

            now = datetime.datetime.utcnow()
            delta = datetime.timedelta(seconds=self.time_difference_to_server)
            now = now + delta

            delta = datetime.timedelta(minutes=1)
            next_minute = (now + delta).replace(microsecond=0, second=0)

            wait_seconds = (next_minute - now).seconds
            wait_microseconds = (next_minute - now).microseconds / 1_000_000
            wait = wait_seconds + wait_microseconds

            if int(wait) % 5 == 0 and (wait - int(wait)) < self.sleep_time:
                logger.debug(
                    f'{now} -> {self.next_minute} :: wait: {round(wait, 2)} seconds'
                )

            if now.minute == self.next_minute:
                self._on_full_minute(now.minute)

            if self.expired_intervals:
                logger.debug(
                    f'callback triggered at {datetime.datetime.utcnow()}'
                )

                logger.debug(self.expired_intervals)

                self.callback(self.expired_intervals)
                self.expired_intervals = []
                time.sleep(10)

            else:
                time.sleep(self.sleep_time)

    def set_time_difference_to_server(self, time_difference_in_ms:int):
        print(f'setting time difference to server ({time_difference_in_ms})')
        self.time_difference_to_server = time_difference_in_ms / 1000

    def _on_full_minute(self, minute:int):
        self.next_minute += 1
        if self.next_minute == 60:
            self.next_minute = 0

        if minute == 0:
            minute_intervals = [i for i in VALID_INTERVALS if 'm' in i]
            [self.expired_intervals.append(i) for i in minute_intervals]
            self._on_full_hour()

        else:
            self.expired_intervals.append('1m')

            if minute % 2 == 0 :
                self.expired_intervals.append('2m')

            if minute % 3 == 0 :
                self.expired_intervals.append('3m')

            if minute % 5 == 0:
                self.expired_intervals.append('5m')

            if minute % 15 == 0:
                self.expired_intervals.append('15m')

            if minute % 30 == 0:
                self.expired_intervals.append('30m')

    def _on_full_hour(self):
        hour = datetime.datetime.utcnow().hour

        self.next_hour += 1
        if self.next_hour == 24:
            self.next_hour = 0

        if hour == 0:
            hour_intervals = [i for i in VALID_INTERVALS if 'h' in i]
            [self.expired_intervals.append(i) for i in hour_intervals]
            self._on_full_day()

        else:
            self.expired_intervals.append('1h')

            if hour % 2 == 0:
                self.expired_intervals.append('2h')

            if hour % 4 == 0 :
                self.expired_intervals.append('4h')

            if hour % 6 == 0:
                self.expired_intervals.append('6h')

            if hour % 8 == 0:
                self.expired_intervals.append('8h')

            if hour % 12 == 0:
                self.expired_intervals.append('12h')

    def _on_full_day(self):
        now = datetime.datetime.utcnow()
        self.expired_intervals.append('1d')

        if now.strftime("%A") == 'Monday':
            self.expired_intervals.append('1w')

        if now.day == 1:
            self.expired_intervals.append('1M')

# =============================================================================
class OhlcvObserver:
    """Observer that monitors OHLCV data and notifies subscribers

    This class keeps track of updates to (relevant) OHLCV data.
    Subscribers can register themselves and the symbols (for different
    intervals) that they are interested in. They will be notified every
    time that new data is available.

    NOTE:   For now we download the data from the REST API. This could
    also be done via websockets, but I found this to be  unreliable for
    the long term.
    Maybe this could be changed in the future. We would need a  much
    more complicated logic and error handling for the  benefit of being
    slightly to much faster (much faster because with the REST API we
    sometimes get a '429' response).
    """

    def __init__(self, exchange: str='kucoin'):
        self.exchange = ExchangePublic(exchange=exchange)
        self.exchange_name: str = exchange
        self.crypto_watch = CryptoWatch()
        self.ohlcv_validator = OhlcvValidator()
        self.update_timer = Chronos(callback=self.intervals_have_passed)
        self.update_timer.start()

        self.subscribers : dict = {
           '1m' : None,
           '5m' : None,
           '15m' : None,
           '30m' : None,
           '1h' : None,
           '2h' : None,
           '4h' : None,
           '6h' : None,
           '8h' : None,
           '12h' : None,
           '1d' : None,
           '3d' : None,
           '1W' : None,
           '1M' : None
        }
        self._max_workers = 10

    # -------------------------------------------------------------------------
    def register_subscriber(self, id:str, symbols:Iterable[str], interval:str,
                            callback:Callable):

        subscriber = namedtuple('Subscriber', ('id', 'symbols', 'callback'))

        if not interval in VALID_INTERVALS:
            logger.critical(
                f'cannot add subscriber <{id}>: invalid interval {interval}'
            )
            return

        try:
            if not interval in self.subscribers.keys():
                self.subscribers[interval] = None

            if self.subscribers[interval] is None:
                self.subscribers[interval] = []

            self.unregister_subscriber(id)

            self.subscribers[interval].append(
                subscriber(id=id, symbols=tuple(set(symbols)), callback=callback)
            )

            logger.debug(f'added subscriber <{id}> ({interval} : {symbols})')

        except Exception as e:
            logger.debug(
                f'unable to add subscriber {id} ({interval} : {symbols}) -> {e}'
            )

    def unregister_subscriber(self, id:str):
        logger.debug(f'unregistering subscriber {id}')
        for interval, subscribers_for_interval in self.subscribers.items():
            if subscribers_for_interval:
                for idx, subscriber in enumerate(subscribers_for_interval):
                    if subscriber.id == 'id':
                        del subscribers_for_interval[idx]
                        logger.debug(f'removed subscriber <{id}> ({interval}')

    def intervals_have_passed(self, intervals:List[str]):
        """Callback method for the Chronos class.

        This method is called by the Chronos class that keeps track
        which intervals have ended. This information is then used to
        make the necessary downloads based on the subscribers.

        :param intervals: list of intervals (1m ... 1d)
        :type intervals: List[str]
        """
        requests = []

        for interval in intervals:
            if interval in self.subscribers:
                symbols = self._get_symbols_to_download_for_interval(interval)

                if symbols:
                    [requests.append((s, interval)) for s in symbols]

                    logger.debug(
                        f'going to download OHLCV for {symbols} '\
                            f'in interval {interval}'
                    )

        if requests:
            data = self._download_ohlcv(requests)
            [self._notify_subscribers(i, data) for i in intervals]

    # -------------------------------------------------------------------------
    def _get_symbols_to_download_for_interval(self, interval:str):

        subscribers_for_interval = self.subscribers.get(interval)

        if not subscribers_for_interval:
            return

        symbols_to_download = []

        for subscriber in subscribers_for_interval:
            [symbols_to_download.append(s) for s in subscriber.symbols]

        return tuple(set(symbols_to_download))

    def _download_ohlcv(self, requests: Iterable[Tuple[str, str]]) -> List[dict]:
        """Concurrently downloads OHLCV data for all requests.

        :param requests: Iterable of tuples (symbol, interval)
        :type requests: Iterable[tuple]
        :return: list of donwload results
        :rtype: List[dict]
        """
        results , futures = [], []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers) as executor:

            for r in requests:
                futures.append(
                    executor.submit(
                        self._download_ohlcv_for_one_symbol,
                        symbol=r[0],
                        interval=r[1]
                        )
                    )

            futures, _ = concurrent.futures.wait(futures)
            [results.append(future.result()) for future in futures]

        return results

    def _download_ohlcv_for_one_symbol(self, symbol:str, interval:str) -> dict:
        """_summary_

        :param symbol: _description_
        :type symbol: str
        :param interval: _description_
        :type interval: str
        :return: _description_
        :rtype: dict
        """
        res = {'symbol' : symbol, 'interval' : interval, 'data' : None}
        query_result, start = None, -1 * NUMBER_OF_INTERVALS
        logger.debug(f'downloading OHLCV for {symbol} ({interval})')

        try:
            query_result = self.exchange.get_ohlcv(
                symbol=symbol, interval=interval, start=start, end=None
            )

            logger.debug(query_result['execution time'])

            if query_result.get('success'):
                res['data'] = query_result['message']

        except Exception as e:
            logger.critical(
                f'unable to download ohlcv for {symbol}: {query_result} - {e}'
            )

        if self._is_valid_result(res['data'], interval, start):
            return self._format_data(res, start)

        # if we did not get valid/complete data from the exchange, then
        # use our alternative data source to fetch it
        res['data'] = self._download_from_alternative_data_source(
            symbol=symbol, interval=interval
        )

        return self._format_data(res, start)

    def _is_valid_result(self, ohlcv: pd.DataFrame, interval: str,
                         start: int) -> bool:
        """Validates downloaded OHLCV data.

        The method checks if we got a dataframe from our download
        method and then uses OhlcvValidator to check if any values
        are missing (which happens quite often because of exchange
        maintenance and other reasons).

        :param item: should be a DataFrame with OHLCV data
        :type item: DataFrame
        :param interval: a trading interval ('1m' ... '1d')
        :type interval: str
        :param start: a negative number of intervals that should be
        contained in the DataFrame
        :type start: int
        """
        end = int(
            datetime.datetime.now(datetime.timezone.utc)\
                .replace(second=0, microsecond=0).timestamp()
        )

        if not isinstance(ohlcv, pd.DataFrame):
            logger.error(f'downloaded data came not back as dataframe')
            return False

        if ohlcv.empty:
            logger.error(f'we only got an empty dataframe')
            return False

        missing_values = self.ohlcv_validator.find_missing_rows_in_df(
            df=ohlcv, interval=interval, start=start, end=end
        )

        if missing_values:
            logger.warning(f'missing values for: {missing_values}')
            return False

        return True

    def _format_data(self, data: dict, start: int) -> dict:
        """Cleans and prepares an OHLCV DataFrame"""
        # set the index
        try:
            data['data'].set_index('human open time', inplace=True) # type: ignore
        except Exception as e:
            logger.debug(e)

        # shorten the dataframe for faster processing
        try:
            data['data'] = data['data'][start:].copy()
        except:
            pass

        # drop the columns that we don´t need anyway
        try:
            [data['data'].drop(col, axis=1, inplace=True) \
                for col in ['close time', 'quote volumne']
            ]  # type: ignore
        except:
            pass

        return data

    def _download_from_alternative_data_source(self, symbol:str, interval:str
                                               ) -> pd.DataFrame:
        """Downloads OHLCV data from Cryptowatch API.

        :param symbol: name of the symbol (e.g. 'BTC-USDT')
        :type symbol: str
        :param interval: the trading interval ('1m' ... '1d')
        :type interval: str
        :return: OHLCV data for the symbol (1000 most recent intervals)
        :rtype: pd.DataFrame
        """
        data = self.crypto_watch.get_ohlcv(
            exchange=self.exchange_name, symbol=symbol, interval=interval
        )

        logger.debug(
            f'downloaded data from crpytowatch - rows: ({len(data)})'
        )

        return data

    def _notify_subscribers(self, interval:str, data:List[dict]):

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

        if interval in self.subscribers.keys():

            if not self.subscribers[interval]:
                return

            for recipient in self.subscribers[interval]:
                try:
                    message = [
                        item.copy() for item in data \
                            if item['symbol'] in recipient.symbols \
                                and item['interval'] == interval
                    ]
                except Exception:
                    logger.exception('')
                    message = None

                try:
                    executor.submit(
                        recipient.callback,
                        {'id' : recipient.id, 'data' : message}
                    )
                    logger.debug(f'sent update to {recipient.id}')
                except Exception:
                    logger.exception('')
