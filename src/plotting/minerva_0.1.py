#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd
from pyrsistent import b

from analysis.indicators import Indicators

import sys

# ----------------------------------------------------------------------------

class Minerva:

    def __init__(self):

        self.name = 'MINERVA' 
        self.facecolor = 'antiquewhite'

    # -------------------------------------------------------------------------
    def test_drawing(self, df):

        # ax = plt.gca()
        fig, axes = plt.subplots(2, 1)
        fig.set_dpi(150)

        # set parameters
        ax1 = axes[0]
        ax2 = axes[1]
        
        # set the fontsize for ticks and labels
        self.fontsize = 4
        ax1.tick_params(axis='x', labelsize=self.fontsize)
        ax1.tick_params(axis='y', labelsize=self.fontsize)
        ax2.tick_params(axis='x', labelsize=self.fontsize)
        ax2.tick_params(axis='y', labelsize=self.fontsize)

        ax1.legend(fontsize=self.fontsize)
        ax2.legend(fontsize=self.fontsize)


        # print the two sma values
        columns = [item for item in df.columns if 'sma' in item]

        df.plot(kind='line', y=columns[0], color='blue', linewidth=0.2, ax=ax1)
        df.plot(kind='line', y=columns[1], color='fuchsia', linewidth=0.3, ax=ax1)

        # print the line for column 'capital'  
        df.plot(kind='line', y='rsi.close.14', color='darkred', alpha=0.7, linewidth=0.5, ax=ax2)

        # finally ... show the plot
        plt.show()        

        return

    # ------------------------------------------------------------------------- 
    def draw_detailed_chart(self, df: pd.DataFrame):
        
        # make sure that the columns we need are numeric values
        columns = ['Open', 'High', 'Low', 'Close',
                   'SMA7', 'SMA21', 'mom', 'mom.sma.3','mom.sma.7',
                    'rsi.close.14', 'sl.current', 'b.value', 'buy.price', 'sell.price',
                    'stoch.close.k', 'stoch.close.d',
                    'stoch.rsi.k', 'stoch.rsi.d', 'stoch.rsi.diff']

        for col in columns: 
            if col in df.columns: df[col] = df[col].apply(pd.to_numeric)

        trend_signal = 's.trnd'

        if trend_signal == 's.adx':
            
            if not 's.adx' in df.columns:
                from analysis.analysts import AverageDirectionalIndexAnalyst
                
                a = AverageDirectionalIndexAnalyst()
                df = a.get_signal(df=df, lookback=30)
            
            df.loc[df[trend_signal] == 1, 'Close.up'] = df['Close']
            df.loc[df[trend_signal] == -1, 'Close.down'] = df['Close']
            df['Close.flat'] = df['Close']
            
        elif trend_signal == 's.trnd':    
            
            if not 's.trnd' in df.columns:

                from analysis.analysts import TrendAnalyst
                
                a = TrendAnalyst()
                df = a.get_signal(df=df)
            
        df['s.state'] = 'flat'
        df.loc[df[trend_signal] == 1, 's.state'] = 'bull'
        df.loc[df[trend_signal] == -1, 's.state'] = 'bear'
            
                    
        # ---------------------------------------------------------------------
        # if not 'stoch.rsi.k' in df.columns:
        #     from analysts import StochRsiAnalyst
            
        #     a = StochRsiAnalyst()
        #     df = a.get_signal(df=df) 
            
        if not 'stoch.close' in df.columns:
            from analysis.analysts import StochasticAnalyst
            
            a = StochasticAnalyst()
            df = a.get_signal(df=df) 
        
        if not 'cci' in df.columns:
            from analysis.analysts import CommodityChannelAnalyst
            
            a = CommodityChannelAnalyst()
            a.overbought, a.oversold = 100, -100
            df = a.get_signal(df=df, period=20)
        
        
        # ---------------------------------------------------------------------
        self.fontsize = 3

        # ax = plt.gca()
        fig, axes = plt.subplots(6, 1,
                                 figsize=(12, 5), 
                                 sharex=True, 
                                 gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [9, 1, 1, 3, 1, 2]})
        fig.set_dpi(160)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]

        fig.patch.set_facecolor('#0f000f')
        face_color = '#18011f'
        tick_color = '#5f016A'
        color_position = '#5f016A'
        color_buy = 'chartreuse'
        color_sell = 'red'
        
        start_index = df.first_valid_index()
        end_index = df.last_valid_index()
        

        for ax in axes:
            ax.set_facecolor(face_color)
            ax.tick_params(axis='x', labelsize=self.fontsize, colors=tick_color)
            ax.tick_params(axis='y', labelsize=self.fontsize, colors=tick_color)

        # ax1.set_yscale('log')
        alpha = 0.5
        signal_alpha = 0.3
        x = 'Human Open Time'
        
        # ---------------------------------------------------------------------
        # first subplot
        lw = 50 / (len(df)/10)
        if 'p.actv' in df.columns:
            for x_ in range(len(df)):
                if df.iloc[x_]['p.actv'] == True:
                    ax1.axvline(x=x_, color=color_position, alpha=0.2, 
                                linewidth=lw, linestyle='-')

        if 'cool.off' in df.columns:
            for x_ in range(len(df)):
                if df.iloc[x_]['cool.off'] == True:
                    ax1.axvline(x=x_, color=color_sell, alpha=0.2, 
                                linewidth=lw, linestyle='-')
                    
                    
        # plot CLOSE price and MOVING AVERAGES in first subplot
        columns = [item for item in df.columns if item.split('.')[0] == 'sma']
        
        if len(columns) == 0:
            columns = [item for item in df.columns if item.split('.')[0] == 'ewm']        

        try:
            df.plot(kind='line', x=x, y=columns[0], color='red', linewidth=0.1, \
                    alpha=0.8, ax=ax1)
            df.plot(kind='line', x=x, y=columns[1], color='orange', linewidth=0.1, \
                    alpha=0.8, ax=ax1)
        except: pass

        if 'ewm.63' in df.columns:
            df.plot(kind='line', x=x, y='ewm.63', color='lightblue', linewidth=0.5, \
                    linestyle='dotted', alpha=0.5, ax=ax1)

        # plot STOP LOSS levels
        if 'sl.current' in df.columns:
            df.plot(kind='line', x=x, y='sl.current', color='violet', 
                    linewidth=0.1, alpha=0.8, drawstyle="steps-mid", ax=ax1)

        # plot KELTNER CHANNEL
        if 'kc.upper' in df.columns:
            df.plot(kind='line', x=x, y='kc.upper', color='#ffab0f', 
                    linestyle='dotted', linewidth=0.2, ax=ax1)

            df.plot(kind='line', x=x, y='kc.lower', color='#ffab0f', 
                    linestyle='dotted', linewidth=0.2, ax=ax1)

            df.plot(kind='line', x=x, y='kc.mid', color='#ffab0f', 
                    linestyle='-', linewidth=0.2, ax=ax1)


        # ---------------------------------------------------------------------
        # plot all the candles - color coded for market state
        
        #define width of candlestick elements
        width = 0.4
        width2 = 0.05
        
        #define up and down prices
        up = df
        # down = df[df.Close < df.Open].astype(float, copy=True)


        #define colors to use
        bull = '#AAFFAA'
        bear = '#FFAAAA'
        flat = 'grey'
        alpha= 0.7
        
        for idx in range(start_index+1, end_index+1):
            x_ = idx - start_index
            _open, _high = up['Open'].iloc[x_], up['High'].iloc[x_]
            _low, _close = up['Low'].iloc[x_], up['Close'].iloc[x_]
            state = up['s.state'].iloc[x_]
            
            if state == 'bull':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=bull, alpha=alpha)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bull, alpha=alpha)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bull, alpha=alpha)  

                else:            
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bull, alpha=alpha)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bull, alpha=alpha)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bull, alpha=alpha)
                    
            elif state == 'bear':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=bear, alpha=alpha)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bear, alpha=alpha)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bear, alpha=alpha)  

                else:            
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bear, alpha=alpha)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bear, alpha=alpha)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bear, alpha=alpha)

            elif state == 'flat':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=flat, alpha=alpha)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=flat, alpha=alpha)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=flat, alpha=alpha)  

                else:            
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=flat, alpha=alpha)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=flat, alpha=alpha)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=flat, alpha=alpha)            

        # ---------------------------------------------------------------------
        df.plot(x=x, y='buy.price', alpha=0.7, marker='^', color=color_buy, 
                markersize=1, ax=ax1)
        df.plot(x=x, y='sell.price', alpha=0.7, marker='v',color=color_sell, 
                markersize=1, ax=ax1)
        
        alpha = 0.5 
        
        # plot up prices
        # ax1.bar(x, height=abs(up.Close-up.Open), width=width, bottom=up.Open, color=col1)
        # ax1.bar(x, height=abs(up.High-up.Close), width=width2, bottom=up.Close, color=col1)
        # ax1.bar(x, height=abs(up.Low-up.Open), width=width2, bottom=up.Open, color=col1)

        #plot down prices
        # ax1.bar(x=x, height=down.Open-down.Close, width=width, bottom=down.Open, color=col2)
        #ax1.bar(x=x, height=down.High-down.Open, width=width2, bottom=down.Open, color=col2)
        #ax1.bar(x=x, height=down.Low-down.Close, width=width2, bottom=down.Close ,color=col2)


        # ---------------------------------------------------------------------
        # second subplot
        if 'stoch.close' in df.columns:
            
            if 's.stc' in df.columns:
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.stc'] == 1: 
                        ax2.axvline(x_, color=color_buy, linewidth=0.2, alpha=signal_alpha)
                    if df.loc[idx, 's.stc'] == -1: 
                        ax2.axvline(x_, color=color_sell, linewidth=0.2, alpha=signal_alpha)
                    
            df.plot(x=x, y='stoch.close.k',kind='line', color='yellow', alpha=alpha, 
                    linewidth=0.1, ax=ax2)
            df.plot(x=x, y='stoch.close.d',kind='line', color='orange', alpha=alpha, 
                    linewidth=0.1, ax=ax2)
            
                    
        # ---------------------------------------------------------------------
        # third subplot
        if 'stoch.rsi.k' in df.columns:
            
            if 's.stc.rsi' in df.columns:
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.stc.rsi'] == 1: 
                        ax3.axvline(x_, color=color_buy, linewidth=0.2, alpha=signal_alpha)
                    if df.loc[idx, 's.stc.rsi'] == -1:
                        ax3.axvline(x_, color=color_sell, linewidth=0.2, alpha=signal_alpha)

            ax3.set_ylim(0,100)
            df.plot(kind='line', x=x, y='stoch.rsi.k', color='yellow' , alpha=0.5, linewidth=0.1, ax=ax3)
            df.plot(kind='line', x=x, y='stoch.rsi.d', color='orange' , alpha=0.5, linewidth=0.1, ax=ax3)
            ax3.axhline(y = 90, color = tick_color, linestyle = 'dotted', linewidth=0.5)
            ax3.axhline(y = 10, color = tick_color, linestyle = 'dotted', linewidth=0.5)            

        else:
            ax3.set_ylim(0,1)
            df.plot(kind='line', x=x, y='b.drawdown', color='yellow' , alpha=0.5, linewidth=0.1, ax=ax3)
            df.plot(kind='line', x=x, y='hodl.drawdown', color='orange' , alpha=0.5, linewidth=0.1, ax=ax3)

        # ---------------------------------------------------------------------
        # fourth subplot
        if 'mom' in df.columns:
            columns = [col for col in df.columns if 'mom.' in col]
            ax2.set_ylim(-10.5, 10.5)

            # df.plot(x=x, y='mom',kind='line', color='dimgrey', linewidth=0.1, ax=ax4)
            
            try:
                df.plot(x=x, y=columns[0], kind='line', color='red', alpha=0.8, 
                        linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)

            try:
                df.plot(x=x, y=columns[1], kind='line', color='orange', 
                        alpha=0.8, linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)

            try:
                df.plot(x=x, y=columns[2], kind='line', color='antiqeuwhite', 
                        alpha=0.8, linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)
            
            if 's.mom' in df.columns:
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.mom'] == 1: ax4.axvline(x_, color=color_buy, alpha=signal_alpha, linewidth=0.2)
                    if df.loc[idx, 's.mom'] == -1: ax4.axvline(x_, color=color_sell, alpha=signal_alpha, linewidth=0.2)
                    
            # plot lines for threshholds of momentum indicator
            threshhold_high = 9
            ax4.axhline(y=0, color='darkgrey', linestyle='dotted', linewidth=0.1)
            ax4.axhline(y=threshhold_high, color='green', linestyle='dotted', linewidth=0.1)
            ax4.axhline(y=threshhold_high*-1, color='green', linestyle='dotted', linewidth=0.1)

        elif ('rsi.close.7' or 'rsi.close.14') in df.columns:
            rsi_columns = [col for col in df.columns if 'rsi.close' in  col]
            df.plot(x=x, y=rsi_columns[0], kind='line', color='red', alpha=0.8, 
                        linewidth=0.1, ax=ax4)            
        
        elif 'stoch.rsi.diff' in df.columns:
            df.plot(x=x, y='stoch.rsi.diff', kind='line', color='red', alpha=0.8, 
                        linewidth=0.1, ax=ax4)
            ax2.axhline(y=0, color='darkgrey', linestyle='dotted', linewidth=0.1)
        
        elif 'ATR percent' in df.columns:
            df.plot(kind='line', x=x, y='ATR percent', color='orange', alpha=0.3, linewidth=0.5, ax=ax4)

        # ---------------------------------------------------------------------
        # fifth subplot
        if 's.cci' in df.columns:
            for idx in df.index:
                x_ = idx - start_index
                if df.loc[idx, 's.cci'] == 1: 
                    ax5.axvline(x_, color=color_buy, linewidth=0.2, alpha=signal_alpha)
                if df.loc[idx, 's.cci'] == -1:
                    ax5.axvline(x_, color=color_sell, linewidth=0.2, alpha=signal_alpha)

        df.plot(x=x, y='cci', kind='line', color='orange', alpha=0.5, 
                linestyle='solid', linewidth=0.1, ax=ax5)        
        
        
        
        # ---------------------------------------------------------------------
        # sixth subplot
        df.plot(kind='line', x=x, y='b.value', color='darkorange', alpha=0.3, linewidth=0.25, ax=ax6)
        
        if 'hodl.value' in df.columns:
            # ax7 = ax6.twinx()
            hodl_color = tick_color
            
            df.plot(kind='line', x=x, y='hodl.value', color='darkred', alpha=0.8, 
                    linestyle='dotted', linewidth=0.2, ax=ax6)
            
            ax6.tick_params(axis ='y', labelcolor=hodl_color, labelsize=self.fontsize)


        # ---------------------------------------------------------------------
        # ax1.grid()
        for ax_ in axes:
            ax_.grid(which='both', linewidth=0.1, linestyle='dotted', 
                     color=tick_color, alpha=1)
            ax_.legend(fontsize=self.fontsize)

        # ax7.legend(fontsize=self.fontsize)
        # ax7.legend().set_visible(False)

        # title = f'{PAIR} ({INTERVAL}) from {START} to {END}'
        # plt.title(title, y=1.0)
        
        # this controls the number/spacing of ticks on the plot
        number_of_major_ticks = 20
        number_of_minor_ticks = 5
        ax1.xaxis.set_major_locator(MultipleLocator(number_of_major_ticks))
        ax1.xaxis.set_minor_locator(MultipleLocator(number_of_minor_ticks))

        ax1.yaxis.set_major_locator(MultipleLocator(df['Close'].max()/10))
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        
        # fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()        

        return



# ============================================================================= 
# draw a chart for the resulting data(frame) of ORACLE
class OracleChart(Minerva):

    def __init__(self, df):

        self.df = df

        self.facecolor = '' #'antiquewhite'
        self.linecolors = ['blue', 'fuchsia', 'darkred', 'green', 'darkslategray']
        self.signalcolor = ['green']

        self.dpi = 150
        self.fontsize = 4

    # ------------------------------------------------------------------------- 
    def draw_signal_chart(self, cols_subplot_1, cols_subplot_2, slope_cols, signal_cols):

        df = self.df

        # prepare dataframe
        # df.set_index('Human Open Time', drop=True, inplace=True)
        df.dropna(inplace=True)
        for col in slope_cols: df[col] = df[col].astype(float)
        for col in signal_cols: df[col] = df[col].astype(int)
        
        if not 'kc.upper' in df.columns:
            i = Indicators()
            df = i.keltner_channel(df=df)


        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(5, 1, 
                                 sharex=True, 
                                 figsize=(12, 5.5),
                                 gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [6, 1, 1, 1, 1]
                                              })

        fig.patch.set_facecolor('#0f000f')
        self.face_color = '#0A0D64'
        # self.face_color = 'antiquewhite'
        tick_color = '#5f016A'
        color_buy = '#5f016A'
        color_sell = '#fa4224'

        fig.set_dpi(self.dpi)
        # fig.patch.set_facecolor('#996600')

        title = f'signals'
        plt.title(title)

        ax_all = axes[0]
        ax_trend = axes[1]
        ax_stoch = axes[2]
        ax_rsi = axes[3]
        ax_mom = axes[4]


        x = 'Human Open Time'

        # ------------------------------------------------------------------------- 
        # plot vertical lines for buy signals
        lw = 0.15
        alpha = 0.5
        start_index = df.first_valid_index()
        end_index = df.last_valid_index()
        
        for idx in df.index:

            color = 'springgreen'
            x = idx
            try:
                if df.loc[idx, 's.all'] > 0:
                    ax_all.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass

            if df.loc[idx, 's.trnd'] > 0:
                ax_trend.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            if df.loc[idx, 's.rsi'] > 0:
                ax_rsi.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            if df.loc[idx, 's.ma.x'] > 0:
                ax_trend.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            try:
                if df.loc[idx, 's.stc.rsi'] > 0:
                    ax_stoch.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass

            try:
                if df.loc[idx, 's.mom'] > 0:
                    ax_mom.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass

            color = 'orangered'

            try:
                if df.loc[idx, 's.all'] < 0:
                    ax_all.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass

            if df.loc[idx, 's.trnd'] < 0:
                ax_trend.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            if df.loc[idx, 's.rsi'] < 0:
                ax_rsi.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            if df.loc[idx, 's.ma.x'] < 0:
                ax_trend.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')

            try:
                if df.loc[idx, 's.stc.rsi'] < 0:
                    ax_stoch.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass

            try:
                if df.loc[idx, 's.mom'] < 0:
                    ax_mom.axvline(x, color=color, alpha=alpha, linewidth=lw,   linestyle='-')
            except: pass
        # ------------------------------------------------------------------------- 
        # plot Close price and moving averages in first and second subplot
        alpha_high = 1
        alpha_low = 0.5
        alpha = alpha_low
        thin = 0.1
        thinner = 0.2
        thicker = 0.3
        SHORT_SMA = 'sma.7'
        LONG_SMA = 'sma.21'

        df.plot(kind='line', y='Close', color='fuchsia', alpha=alpha_high, linewidth=thicker, ax=ax_all)
        df.plot(kind='line', y=SHORT_SMA, color='blue', linewidth=thinner, alpha=alpha_high, ax=ax_all)
        df.plot(kind='line', y=LONG_SMA, color='grey', linewidth=thinner, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='kc.upper', color='grey', 
                linestyle='dotted', linewidth=0.2, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='kc.lower', color='grey', 
                linestyle='dotted', linewidth=0.2, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='Close', color='fuchsia', alpha=alpha_high, linewidth=thicker, ax=ax_trend)
        df.plot(kind='line', y=SHORT_SMA, color='blue', linewidth=thinner, alpha=alpha_high, ax=ax_trend)
        df.plot(kind='line', y=LONG_SMA, color='grey', linewidth=thinner, alpha=alpha_high, ax=ax_trend)
        # ------------------------------------------------------------------------- 
        # subplot: momentum (sma)
        df.plot(kind='line', y='mom.sma.3', color='lightblue', alpha=alpha, linewidth=thinner, ax=ax_mom)
        df.plot(kind='line', y='mom.sma.9', color='orange', alpha=alpha, linewidth=thinner, ax=ax_mom)
        df.plot(kind='line', y='mom', color='antiquewhite', linestyle='dotted', alpha=alpha, linewidth=0.5, ax=ax_mom)
        ax_mom.axhline(y=0, color='grey', linestyle='dotted', alpha=alpha, linewidth=0.5)

        # plot lines for mom threshholds
        threshhold_low = -9
        threshhold_high = 9
        ax_mom.axhline(y=threshhold_low, color='green', linestyle='dotted', linewidth=0.5)
        ax_mom.axhline(y=threshhold_high, color='green', linestyle='dotted', linewidth=0.5)

        # ------------------------------------------------------------------------- 
        # third subplot
        # ax_rsi.set_ylim(0,100)
        # column = [col for col in df.columns if 'rsi.Close.' in col]
        # df.plot(kind='line', y=column[0], color='lightblue', alpha=alpha, linewidth=thinner, ax=ax_rsi)

        # plot lines for rsi threshholds
        # threshhold_low = 5
        # threshhold_high = 95
        # ax_rsi.axhline(y=threshhold_low, color='green', linestyle='dotted', alpha=alpha, linewidth=0.5)
        # ax_rsi.axhline(y=threshhold_high, color='green', linestyle='dotted', alpha=alpha, linewidth=0.5)

        df.plot(kind='line', y='slope.sma.7', color='lightblue', alpha=alpha, linewidth=thinner, ax=ax_rsi)
        ax_rsi.axhline(y=0, color='green', linestyle='dotted', alpha=alpha, linewidth=0.5)

        # ---------------------------------------------------------------------
        ax_stoch.set_ylim(0,100)
        df.plot(kind='line', y='stoch.rsi.k', color='orange', alpha=alpha_low, linewidth=thin, ax=ax_stoch)
        df.plot(kind='line', y='stoch.rsi.d', color='blue', alpha=alpha_low, linewidth=thin, ax=ax_stoch)

        # plot lines for stoch rsi threshholds
        threshhold_low = 10
        threshhold_high = 90
        ax_stoch.axhline(y=threshhold_low, color='green', linestyle='dotted', alpha=alpha, linewidth=thicker)
        ax_stoch.axhline(y=threshhold_high, color='green', linestyle='dotted', alpha=alpha, linewidth=thicker)




        # ------------------------------------------------------------------------- 
        # set fontsize
        for ax_ in axes:
            ax_.set_facecolor(self.face_color)
            ax_.tick_params(axis='x', labelsize=self.fontsize, color=tick_color)
            ax_.tick_params(axis='y', labelsize=self.fontsize, color=tick_color)
            ax_.legend(fontsize=self.fontsize)
        
        plt.tight_layout()
        plt.show()        

        return



# =============================================================================
class MosesChart:

    def __init__(self, df):

        self.df = df

        # matplotlib attributes
        self.facecolor = 'antiquewhite'
        self.linecolors = ['blue', 'fuchsia', 'darkred', 'green', 'darkslategray']
        self.signalcolor = 'red'

        self.dpi = 150
        self.fontsize = 4

        # self.draw_chart()


    # ------------------------------------------------------------------------- 
    def draw_chart(self):

        df = self.df

        #df.set_index('Human Open Time', drop=True, inplace=True)
        df.dropna(inplace=True)
        
        # ax = plt.gca()
        fig, axes = plt.subplots(2, 1, 
                                 sharex=True, 
                                 figsize=(12, 5),
                                 gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [1, 1]
                                              })
        fig.set_dpi(self.dpi)

        # title = 'STOP LOSS stuff'
        # plt.title(title)

        ax_atr_sl = axes[0]
        ax_pctl_sl = axes[1]

        # ------------------------------------------------------------------------- 
        # plot Open + Low price and ATR stop-loss in FIRST subplot
        x = 'Human Open Time'
        df.plot.scatter(x=x, y='Low', s=[0.1], c='red', alpha=0.5, ax=ax_atr_sl)
        df.plot(kind='line', x=x, y='Open', color='blue', linewidth=0.2, drawstyle="steps-mid", ax=ax_atr_sl)
        df.plot(kind='line', x=x, y='sl.atr', color='fuchsia', linewidth=0.3, drawstyle="steps-mid", ax=ax_atr_sl)
        # df.plot(kind='line', y='Low', color='darkslategray', alpha=0.5, linestyle='dotted', linewidth=0.5, ax=ax_atr_sl)

        # ------------------------------------------------------------------------- 
        # plot Open + Low price and PERCENTILE stop-loss in SECOND subplot
        
        df.plot(kind='line', x=x, y='Low', color='blue', alpha=0.7, linewidth=0.2, ax=ax_pctl_sl)
        df.plot(kind='line', x=x, y='sl.pctl', color='fuchsia', linewidth=0.2, drawstyle="steps-mid", ax=ax_pctl_sl)
        df.plot(kind='line', x=x, y='Open', color='darkslategray', alpha=0.5, linestyle='dotted', linewidth=0.5, ax=ax_pctl_sl)

        # ------------------------------------------------------------------------- 
        # set fontsize
        for ax_ in axes:
            ax_.set_facecolor(self.facecolor)
            ax_.tick_params(axis='x', labelsize=self.fontsize, rotation=90)
            ax_.tick_params(axis='y', labelsize=self.fontsize)
            ax_.legend(fontsize=self.fontsize)

        plt.show()        

        return


    # -------------------------------------------------------------------------
    def draw_plotly_chart(self):

        templates = ["plotly", "plotly_white", "plotly_dark", "ggplot2", 
                     "seaborn", "simple_white", "none"]
        template = 'plotly_dark'

        fig = go.Figure(data=[go.Candlestick(x=self.df['Human Open Time'],
                                             open=self.df['Open'],
                                             high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close']
                                             ),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['sl.l.atr'],
                                          visible=True,
                                          name='SL LONG ATR',
                                          line = {"shape": 'hvh',
                                                  "color" : 'azure'}),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['sl.s.atr'],
                                          visible=True,
                                          name='SL SHORT ATR',
                                          line = {"shape": 'hvh',
                                                  "color" : 'azure'}),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['sl.l.pctl'],
                                          name='SL LONG PCTL',
                                          visible=True,
                                          line = {"shape": 'hvh',
                                                  "color" : 'crimson'},
                                                  hovertext=self.df['sl.s.pctl.pct']),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['sl.s.pctl'],
                                          visible=True,
                                          name='SL SHORT PCTL',
                                          line = {"shape": 'hvh',
                                                  "color" : 'crimson'},
                                          hovertext=self.df['sl.s.pctl.pct']),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['tp.l.atr'],
                                          name='TP LONG ATR',
                                          visible=True,
                                          line = {"shape": 'hvh',
                                                  "color" : 'deeppink'}),
                              go.Scatter(x=self.df['Human Open Time'], 
                                          y=self.df['tp.s.atr'],
                                          name='TP SHORT ATR',
                                          visible=True,
                                          line = {"shape": 'hvh',
                                                  "color" : 'deeppink'}),
                              ])

        fig.update_traces(opacity=0.7, selector=dict(type='scatter'))
        fig.update_traces(line_width=0.5, selector=dict(type='scatter'))
        fig.update_traces(line_width=0.5, selector=dict(type='candlestick'))

        fig.update_layout(template=template, yaxis_type="log", hovermode="x")
        # fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        fig.show()



# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    print(pio.templates)