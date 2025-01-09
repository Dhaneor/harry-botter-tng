#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""
import numpy as np

from analysis.models.market_data import MarketData, MarketDataStore
from staff.hermes import Hermes

hermes = Hermes(exchange='binance')

def prepare_data():
    btc_data = hermes.get_ohlcv('BTCUSDT', '1d', "1000 days ago UTC", "now UTC").get("message")
    eth_data = hermes.get_ohlcv('ETHUSDT', '1d', "1000 days ago UTC", "now UTC").get("message")

    # combine the data into a series of 2D Numpy arrays, one each
    # for 'open', 'high', 'low', 'close', 'volume'

    transformed = {}

    for col in ['open time', 'open', 'high', 'low', 'close', 'volume']:
        d_type = np.int64 if col == 'open time' else np.float32

        key = "timestamp" if col == 'open time' else col
        key = "open_" if key == 'open' else key

        transformed[key] = np.transpose(
            np.array((btc_data[col], eth_data[col]), dtype=d_type)
        )

    return transformed, ['BTCUSDT', 'ETHUSDT']

def test_market_data_store_instance():
    data, _ = prepare_data()

    mds = MarketDataStore(**data)

    assert isinstance(mds, MarketDataStore)
    assert mds.log_returns.shape == mds.close.shape
    assert mds.atr.shape == mds.close.shape
    assert mds.annual_vol.shape == mds.close.shape

    return mds


def test_market_data():
    data, symbols = prepare_data()

    mds = MarketDataStore(**data)
    md = MarketData(mds, symbols)

    # print(md["BTCUSDT"]['open'])
    # print(md['close'])
    # print(md.dataframe.info())

    test = np.random.randint(0, 100_000, size=(999, 1), dtype=np.int32)

    print(md.open > test)
    print(md.interval_in_ms)
    print(md.to_dictionary())


def test_from_random():
    md = MarketData.from_random(30, 3)

    assert isinstance(md, MarketData)

    print(md.dataframe.tail())
    print(md.mds.annual_vol)
    print(md.get_array('close'))


if __name__ == '__main__':
    test_from_random()
