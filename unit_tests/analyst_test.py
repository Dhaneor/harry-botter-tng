#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 01 14:03:23 2021

@author dhaneor
"""

import sys
import os
import time
import pandas as pd
import logging

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../backtest.module/')
# -----------------------------------------------------------------------------
from src.analysis.analysts import *
from src.staff.hermes import Hermes

# =============================================================================
class TestAnalyst:

    def __init__(self, name:str):
    
        self.hermes = Hermes(exchange='binance', mode='backtest')
        
    def get_ohlcv(self, symbol: str, start, end,
                 interval: str) -> pd.DataFrame:
        
        _res = self.hermes.get_ohlcv(
            symbols=symbol, interval=interval,  start=start, end=end
        )
        return _res['message']


# =============================================================================
#                               TEST FUNCTIONS                                #
# =============================================================================
def test_get_data():
    ta = TestAnalyst(name='')
    df = ta.get_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    print(df)

# -----------------------------------------------------------------------------    
def test_chart():
    
    # get data 
    ta = TestAnalyst(name='')
    df = ta.get_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    subplots = {}

    # # -------------------------------------------------------------------------
    # add indicator(s) and signal columns
    # a = RsiAnalyst()
    # a.overbought, a.oversold = 80, 25
    # lookback = 20
    
    # df = a.get_signal(df=df, lookback=lookback)

    # subplots['RSI'] = {'columns' : ['rsi.close.' + str(lookback)],
    #                     'horizontal lines' : [a.overbought, a.oversold],
    #                     'fill' : True,
    #                     'signal' : 's.rsi'}

    # # -------------------------------------------------------------------------
    # a = CommodityChannelAnalyst()
    # a.overbought, a.oversold = 100, -100
    # df = a.get_signal(df=df, period=14, mode=2)

    # subplots['Commodity Channel Index'] = {'columns' : ['cci'],
    #                                         'horizontal lines' : [a.overbought, a.oversold],
    #                                         'fill' : True,
    #                                         'signal' : 's.cci'}

    # -------------------------------------------------------------------------
    # a = MomentumAnalyst()
    # a.short_sma, a.long_sma, a.lookback = 3, 7, 9
    # df = a.get_signal(df=df)

    # columns = [col for col in df.columns if 'mom.sma' in col]

    # subplots['Momentum'] = {'columns' : columns,
    #                         'horizontal lines' : [a.overbought, a.oversold],
    #                         'fill' : True,
    #                         'signal' : 's.mom'}

    # --------------------------------------------------------------------------
    # a = MovingAverageAnalyst()
    # a.get_signal(df=df)
    
    # columns = [col for col in df.columns if ('sma' or 'ewm') in col]

    # subplots['Moving Average'] = {'columns' : columns,
    #                               'horizontal lines' : [],
    #                               'fill' : False,
    #                               'signal' : 's.ma'}
    
    # -------------------------------------------------------------------------
    # a = MovingAverageCrossAnalyst()
    # df['s.ma.x'] = a.get_signal(data=df.close.to_numpy())

    # # -------------------------------------------------------------------------
    # a = StochasticAnalyst()
    # a.overbought, a.oversold = 75, 25
    # df = a.get_signal(df=df)

    # subplots['Stochastic'] = {'columns' : ['stoch.close.d', 'stoch.close.k'],
    #                           'horizontal lines' : [a.overbought, a.oversold],
    #                           'fill' : True,
    #                           'signal' : 's.stc'}

    # # -------------------------------------------------------------------------
    # a = StochRsiAnalyst()
    # a.overbought, a.oversold = 75, 25
    # df = a.get_signal(df=df, method='extremes')

    # subplots['Stoch RSI'] = {'columns' : ['stoch.rsi.d', 'stoch.rsi.k'],
    #                           'horizontal lines' : [a.overbought, a.oversold],
    #                           'fill' : True,
    #                           'signal' : 's.stc.rsi'}

    # # -------------------------------------------------------------------------
    # a = ChoppinessIndexAnalyst()
    # a.get_signal(df=df)
    
    # subplots['Choppiness Index'] = {'columns' : ['ci'],
    #                                 'horizontal lines' : [61.8, 38.2],
    #                                 'fill': True,
    #                                 'signal' : 's.ci'}
 
    # # # -------------------------------------------------------------------------
    # a = TrendAnalyst()
    # a.get_signal(df=df)
    # df['s.state'] = df['s.trnd']
    
    # subplots[''] = {'columns' : [],
    #                             'horizontal lines' : [],
    #                             'fill': False,
    #                             'signal' : 's.trnd'} 
    
    # # -------------------------------------------------------------------------
    # a = WickAndBodyAnalyst()
    # df = a.get_signal(df=df, factor=2, confirmation=True)
    
    # subplots['Wick-and-Body'] = {'columns' : [],
    #                             'horizontal lines' : [],
    #                             'fill': False,
    #                             'signal' : 's.wab'}
    
    # # -------------------------------------------------------------------------
    # a = BreakoutAnalyst()
    # df = a.get_signal(df=df, lookback=14)
    # subplots[a.plot_params['label']] = a.plot_params

    # # -------------------------------------------------------------------------
    # a = TrendyAnalyst()
    # df = a.get_signal(df=df, threshhold=1.5, lookback=24, smoothing=6)
    # subplots[a.plot_params['label']] = a.plot_params

    # -------------------------------------------------------------------------
    # a = TDClopwinPatternAnalyst()
    # df = a.get_signal(df=df)
    
    # -------------------------------------------------------------------------
    # a = KeltnerChannelAnalyst()
    
    # data = {
    #     'o': df.open.to_numpy(),
    #     'h': df.high.to_numpy(),
    #     'l': df.low.to_numpy(),
    #     'c': df.close.to_numpy()
    # }
        
    # res = data=a.get_signal(data_as_dict=data, multiplier=3)
    # df['signal'] = res['s.kc']

    
    # -------------------------------------------------------------------------
    # a = DynamicRateOfChangeAnalyst()
    # df = a.get_signal(df=df, lookback=13, smoothing=21)
    # subplots[a.plot_params['label']] = a.plot_params

    # -------------------------------------------------------------------------
    a = FibonacciTrendAnalyst()
    df = a.get_signal(df=df)
    subplots[a.plot_params['label']] = a.plot_params
    
    # a = BigTrendAnalyst()
    # df = a.get_signal(df=df, period=20, factor=0.0001)
    
    # -------------------------------------------------------------------------
    # a = NoiseAnalyst()
    # df = a.get_signal(df=df, period=21, smoothing=14)
    # subplots[a.plot_params['label']] = a.plot_params

    # -------------------------------------------------------------------------
    # a = DisparityAnalyst()
    # df = a.get_signal(df=df)
    # subplots[a.plot_params['label']] = a.plot_params
    
    # -------------------------------------------------------------------------
    # a = ConnorsRsiAnalyst()
    # a.overbought, a.oversold = 95, 5
    # df = a.get_signal(df=df, rsi_lookback=2, streak_lookback=3, smoothing=1)
    # subplots[a.plot_params['label']] = a.plot_params
    
    
    # -------------------------------------------------------------------------
    # combine the signals 
    # try:
    #     df.rename({'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}, axis=1, inplace=True)
    # except:
    #     pass
    
    # df['human open time'] = pd.to_datetime(df['open time'], unit='s')
    
    sig_cols = [col for col in df.columns if col.split('.')[0] == 's']
    df['signal'], df['position'] = np.nan, np.nan
    
    for col in sig_cols:
        df.signal = df[col]
    
    df['position'] = df.signal.shift().ffill()
    
    # find buy and sell points/prices
    df['buy_at'], df['sell_at'] = np.nan, np.nan
    
    df.loc[(df.position > 0) & ~(df.position.shift() > 0), 'buy_at'] = df.close.shift()
    df.loc[(df.position < 0) & ~(df.position.shift() < 0), 'sell_at'] = df.close.shift()
    df.loc[(df.signal == 0) & (df.position == -1), 'buy_at'] = df.close
    df.loc[(df.signal == 0) & (df.position == 1),'sell_at'] = df.close
    
    try:
        df['s.state'] = df['s.ci']
        df['s.state'].replace(0, 'flat', inplace=True)
        df['s.state'].replace(1, 'bull', inplace=True)
        df['s.state'].replace(-1, 'bear', inplace=True)
    except:
        pass

    # -------------------------------------------------------------------------
    # draw the chart
    df = df[200:].copy()
    
    if df.empty:
        print('No data')
        return
    
    print(df.tail(50))
    print('*' * 200)
    print(subplots)
    
    a.draw_chart(
        df=df, subplots=subplots, color_scheme='day', with_market_state=False
    )


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    
    symbol = 'ADA-USDT'
    interval = '1d'
    start = -1000 # 'June 1, 2022 00:00:00'
    end = 'June 23, 2023 00:00:00'

    test_chart()