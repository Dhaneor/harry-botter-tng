#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module prepares an OHLCV download.

Exchanges do not allow to download OHLCV data for more than a certain
number of periods with one call. If more data is needed, the download
request must be split into multiple calls.

The class in this module determines the exact timestamps for the period(s)
or chunks to be downloaded. The chunk timestamps will/can be used by the
caller to download chunks of the max allowed size sequentially (or in
parallel for faster downloads).

Also some sanity checks are done. For instance, if a start date was
given that lies way in the past,

__author__ = @dhaneor
__copyright__ = Copyright 2022
__version__ = 1.0
__maintainer__ = @dhaneor
__email__ = crpytodude23 at protonmail
__status__ = Dev
"""
import logging
from typing import Union, Dict
from collections import OrderedDict
from functools import lru_cache

from src.util.timeops import (  # noqa: E402
    interval_to_milliseconds,
    unix_to_utc,
    get_start_and_end_timestamp,
)

logger = logging.getLogger("main.ohlcv_download_prepper")
logger.setLevel(logging.INFO)

INTERVALS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class OhlcvDownloadPrepper:
    """This class prepares an OHLCV download.

    The class is not used on its own, but for composition into
    exchange classes.

    It determines the exact timestamps for the period to be downloaded
    and also for the chunks. The chunk timestamps will/can be used by
    the caller to download chunks of the max allowed size in parallel
    for faster downloads.

    Also some sanity checks are done. For instance, if a start date was
    given that lies way in the past, it tries to get the listing date
    and adjusts the time period accordingly.
    """

    def __init__(self):
        """Initialize the Download Prepper with a working connection.

        :param client: a working Client object for the exchange
        :type client: _type_
        """
        logger.debug("Initializing Download Prepper")
        self.listing_dates: Dict[str, int] = {}
        self.limit: int = 1000  # max allowed size for one request

    def _prepare_request(
        self, symbol: str, interval: str, start: Union[str, int], end: Union[str, int]
    ) -> dict:
        """Prepare the download of OHLCV data.

        This method determines timestamps (UTC) from the given values
        for 'start' and 'end' and prepares chunks to be downloaded as
        Kucoin is retricting us to a limit of 1500 klines for one
        request. Getting the chunks beforehand is necessary for parallel
        download using threads.

        :param symbol: name of symbol
        :type symbol: str
        :param interval: interval like '1m' .. '4h' ... '1d'
        :type interval: str
        :param start: start time as date or unix timestamp (UTC)
        or number of epochs given as negative number
        :type start: Union[str, int], optional
        :param end: end time as date or unix timestamp (UTC)
        :type end: Union[str, int], optional
        :return: _description_
        :rtype: dict

        .. code:: python
            OrderedDict([('symbol', 'BCHSV-USDT'),
                        ('interval', '1h'),
                        ('start', '2017-01-01'),
                        ('start_ts', 1514764800),
                        ('end', '2021-01-01 12:00:00'),
                        ('end_ts', 1609502400),
                        ('number of chunks', 27),
                        ('chunks', [(1514764800, 6911164799)]),
                        ('results', None)])
        """
        # check for valid interval and return error message not valid
        if interval not in INTERVALS.keys():
            raise ValueError(f"Invalid interval: {interval}")

        # get timestamps for start and end of time period
        start_ts, end_ts = self._get_timestamps_for_period(
            symbol=symbol, start=start, end=end, interval=interval
        )

        # get the number of klines we need to downlaod
        number_of_chunks = self._get_number_of_chunks(start_ts, end_ts, interval)

        # get the chunks (periods for every download step)
        chunks = self._get_chunk_periods(
            start_ts=start_ts, end_ts=end_ts, interval=interval
        )

        return OrderedDict(
            [
                ("symbol", symbol),
                ("interval", interval),
                ("start", unix_to_utc(start_ts)),
                ("start_ts", start_ts),
                ("end", unix_to_utc(end_ts)),
                ("end_ts", end_ts),
                ("number of chunks", number_of_chunks),
                ("chunks", chunks),
                ("results", None),
            ]
        )

    def _get_chunk_periods(self, start_ts: int, end_ts: int, interval: str):
        limit = self.limit
        steps = []
        step_start = start_ts

        while not step_start >= end_ts:
            step_end = int(step_start + limit * INTERVALS[interval] - 1)
            step_end = end_ts if step_end > end_ts else step_end
            steps.append((step_start, step_end))
            step_start = step_end + 1

        return steps

    def _get_number_of_chunks(self, start: int, end: int, interval: str):
        _delta = end - start
        _max = interval_to_milliseconds(interval) / 1000 * self.limit
        epochs, partial = divmod(_delta, _max)

        return int(epochs) if partial == 0 else int(epochs + 1)

    def _get_timestamps_for_period(
        self, symbol: str, start: Union[str, int], end: Union[str, int], interval: str
    ) -> tuple:

        start_ts, end_ts = get_start_and_end_timestamp(
            start=start, end=end, interval=interval, unit="seconds", verbose=False
        )

        # Kucoin does not check if the start timestamp lies way before
        # the symbol was listed and does not adjust that (like Binance
        # does). So we need to get the listing date first and then
        # adjust start_ts here. otherwise we would make useless calls
        # to the API with empty results (and this would be a lot of them,
        # if the interval is short).
        # But we only need to do this if we risk more than one or two
        # empty results as otherwise this would be one API call that
        # can be avoided if the interval is long or if the number of
        # epochs to download is low anyway.
        number_of_epochs = self._get_number_of_chunks(
            start=start_ts, end=end_ts, interval=interval
        )

        if number_of_epochs > 5:

            listing_date = self._get_listing_date(symbol, interval)
            ld_human = unix_to_utc(listing_date)
            start_ts_human = unix_to_utc(start_ts)

            logger.debug(
                "Listing date: %s %s, start: %s",
                ld_human, type(ld_human), start_ts_human,
            )

            # earliest_valid_ts = max(
            #     start_ts, listing_date, utc_to_unix("January 01, 2017 00:00:00")
            # )

        # convert timestamp to seconds if it was given in milliseconds
        start_ts = int(start_ts / 1000) if len(str(start_ts)) > 10 else start_ts
        end_ts = int(end_ts / 1000) if len(str(end_ts)) > 10 else end_ts

        logger.debug("Start timestamp: %s -> %s", start, start_ts)
        logger.debug("End timestamp: {end=} -> {end_ts}", end, end_ts)

        return start_ts, end_ts

    @lru_cache
    def _get_listing_date(self, symbol: str, interval: str = "1d") -> int:
        logger.debug("getting listing date for: %s (%s)", symbol, interval)

        try:
            return self.listing_dates[symbol]
        except KeyError:
            pass

        res = self._get_earliest_valid_timestamp(symbol=symbol, interval=interval)

        logger.info(f"got listing date from API: {res}")

        if isinstance(res, str):
            return int(res)
        elif isinstance(res, dict):
            if "success" in res and res["success"]:
                return int(res["message"])

        if isinstance(res, int):
            self.listing_dates[symbol] = res
            return res

        raise Exception(f"unable to get listing date for {symbol} ({interval})")
