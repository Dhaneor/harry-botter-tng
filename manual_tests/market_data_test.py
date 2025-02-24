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
    md = MarketData.from_random(300, 1, 0.025)

    assert isinstance(md, MarketData)

    df = md.dataframe.round(3)

    print(df.tail(20))

    print(f"scale factor min: {df[:]["signal_scale"].min()}")
    print(f"scale factor max: {df[:]["signal_scale"].max()}")

    md.plot()


def test_from_parameters():
    exchange = "binance"
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    interval = "4h"
    start = "2024-02-01"
    end = "2025-01-31"

    md = MarketData.from_parameters(
        exchange=exchange, symbols=symbols, interval=interval,
        start=start, end=end
    )

    print(len(md))
    print(md.number_of_assets)

    df = md.dataframe

    print(df)


def test_get_array():
    md = MarketData.from_random(30, 3, 0.05)
    print(md.symbols)

    symbol_1 = md.symbols[0]
    symbol_2 = md.symbols[1]

    assert isinstance(md.get_array('close'), np.ndarray)
    # assert np.array_equal(md.get_array('close'), md.mds.close)

    # assert np.array_equal(md.get_array('close', symbol=symbol_2), md.mds.close[:, 1])

    # assert np.array_equal(md.get_array('open', symbol=symbol_2), md.mds.open_[:, 1])
    # assert np.array_equal(md.get_array('high', symbol=symbol_1), md.mds.high[:, 0])
    # assert np.array_equal(md.get_array('low', symbol=symbol_1), md.mds.low[:, 0])


if __name__ == '__main__':
    test_from_parameters()
