#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import asyncio
import time
from random import random

from analysis.strategy.definitions import breakout  # noqa: E402, F401
from staff.hermes import Hermes  # noqa: E402, F401
from analysis.strategy.exit_order_strategies import *  # noqa: E402, F401, F403
from util import get_logger

logger = get_logger('main')

h = Hermes(exchange='kucoin', mode='live')
s = breakout()
s.symbol = 'BTC-USDT'
s.interval = '1d'


def get_data():
    res = h.get_ohlcv(
        symbols='BTC-USDT', interval='1d', start=-1000, end='now UTC'
    )

    if res['success']:
        return res['message']


async def test_process_kline_update(update: dict):
    await s.process_kline_update(update)


async def main():
    data = get_data()

    if data is not None:
        base = data[:100][:]
        updates = data[100:][:]
        s.ohlcv = base
        s.ohlcv.rename(columns={'quote asset volume': 'quote volume'}, inplace=True)
        s.ohlcv.drop(['human open time', 'volume', 'close time'], axis=1, inplace=True)

        for row in updates.iterrows():
            update = row[1].to_dict()
            pub_msg = {
                'subject': 'candles',
                'topic': 'BTC-USDT_1d',
                'type': 'add',
                'symbol': 'BTC',
                'interval': '1d',
                'data': {
                    'open time': update['open time'],
                    'open': update['open'],
                    'high': update['high'],
                    'low': update['low'],
                    'close': update['close'],
                    'volume': update['volume'],
                    'quote volume': update['quote asset volume'],
                },
                'time': update['close time'],
                'reveived_at': time.time() - random() / 10
            }

            await test_process_kline_update(pub_msg)
            await s.push_it()

        print(s.ohlcv.tail(49))


# =============================================================================
if __name__ == '__main__':
    asyncio.run(main())
