#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import StrMethodFormatter, NullFormatter, \
    LogFormatter, SymmetricalLogLocator, ScalarFormatter, \
    FormatStrFormatter

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import datetime as dt

import sys, os
from pprint import pprint


# ----------------------------------------------------------------------------

class Minerva:

    def __init__(self, color_scheme:str='night'):

        self.name = 'MINERVA'
        self.df: pd.DataFrame = None
        self._set_colors(color_scheme)
        

    # ------------------------------------------------------------------------- 
    def _set_colors(self, color_scheme:str):
        
        # import the styles as defined in the (mandatory) file mpl_schemes.py
        # which must reside in the same directory
        from  plotting.mpl_styles import schemes as s
        allowed = [scheme for scheme in s.keys()]
        
        # set color scheme to 'day' (=default) if the given name for the color 
        # scheme is not defined in mpl_styles  
        color_scheme = color_scheme if color_scheme in allowed else 'day'

        # set all colors from the imported color scheme        
        self.canvas = s[color_scheme]['canvas']
        self.background = s[color_scheme]['background']
        self.tick = s[color_scheme]['tick']
        
        self.bull = s[color_scheme]['bull']
        self.bear = s[color_scheme]['bear']
        self.flat = s[color_scheme]['flat']
        
        self.buy = s[color_scheme]['buy']
        self.sell = s[color_scheme]['sell']
        self.position = s[color_scheme]['position']
        
        self.line1 = s[color_scheme]['line1']
        self.line2 = s[color_scheme]['line2']
        self.line3 = s[color_scheme]['line3']
        self.line4 = s[color_scheme]['line4']
        
        self.channel = s[color_scheme]['channel']
        self.channel_bg = s[color_scheme]['channel_bg']
        
        self.grid = s[color_scheme]['grid']
        self.hline = s[color_scheme]['hline']
        
        # make a list of colors/alpha values for lines for easy looping
        self.line_colors = [self.line1[0], self.line2[0], self.line3[0], self.line4[0]]
        self.line_alphas = [self.line1[1], self.line2[1], self.line3[1], self.line4[1]]
  
    def _columns_to_numeric(self, df:pd.DataFrame) -> pd.DataFrame:
        
        '''
        Make sure that the columns we need are numeric values ...
        '''
        columns = ['open', 'high', 'low', 'close',
                   'SMA7', 'SMA21', 'mom', 'mom.sma.3','mom.sma.7',
                    'rsi.close.14', 'sl.current', 'b.value', 'buy.price', 'sell.price',
                    'stoch.close.k', 'stoch.close.d',
                    'stoch.rsi.k', 'stoch.rsi.d', 'stoch.rsi.diff']

        for col in columns: 
            if col in df.columns: df[col] = df[col].apply(pd.to_numeric)
            
        return df
    
    def _add_trend_signal(self, df:pd.DataFrame, trend_signal:str='s.trnd') -> pd.DataFrame:
        
        '''
        This function checks if we have a trend signal (bull/bear/flat) in the
        DataFrame and adds it if necessary 
        
        Possible trend signals (= their columns) are:
        
        's.adx' :   Average Directional Index (ADX) indicator   
        's.trnd' :  my own indicator for market state (this is the default)  
        '''

        if trend_signal == 's.adx':
            
            if not 's.adx' in df.columns:
                from analysis.analysts import AverageDirectionalIndexAnalyst
                
                a = AverageDirectionalIndexAnalyst()
                df = a.get_signal(df=df, lookback=30)
            
            df.loc[df[trend_signal] == 1, 'Close.up'] = df['close']
            df.loc[df[trend_signal] == -1, 'Close.down'] = df['close']
            df['Close.flat'] = df['close']
            
        elif trend_signal == 's.trnd':    
            
            if not 's.trnd' in df.columns:

                from analysis.analysts import TrendAnalyst
                
                a = TrendAnalyst()
                df = a.get_signal(df=df)
            
        df['s.state'] = 'flat'
        df.loc[df[trend_signal] == 1, 's.state'] = 'bull'
        df.loc[df[trend_signal] == -1, 's.state'] = 'bear'        

        return df
        
    # ------------------------------------------------------------------------- 
    def draw_detailed_chart(self, df: pd.DataFrame):
        
        df = self._columns_to_numeric(df)
        df = self._add_trend_signal(df=df, trend_signal='s.trnd')

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

        fig, axes = plt.subplots(6, 1,
                                 figsize=(12, 5), 
                                 sharex=True,
                                 linewidth = 0.5, 
                                 gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [8, 1, 1, 1, 1, 2]})
        fig.set_dpi(160)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]

        fig.patch.set_facecolor(self.canvas)
        face_color = self.background
        tick_color = self.tick[0]
        color_position = self.position[0]
        color_buy = self.buy[0]
        color_sell = self.sell[0]
        bull = 'green' #self.bull[0]
        bear = self.bear[0]
        flat = self.flat[0]
        line1 = self.line1[0]
        line2 = self.line2[0]
        line3 = self.line3[0]
        line4 = self.line4[0]
        channel = self.channel[0]
        channel_bg = self.channel_bg[0]
        grid = self.grid[0]
        
        alpha_position = self.position[1]
        alpha_buy = self.buy[1]
        alpha_sell = self.sell[1]
        alpha_bull = self.bull[1]
        alpha_bear = self.bear[1]
        alpha_flat = self.flat[1]
        alpha_line1 = self.line1[1]
        alpha_line2 = self.line2[1]
        alpha_line3 = self.line3[1]
        alpha_line4 = self.line4[1]
        alpha_grid = self.grid[1] 
        alpha_channel_bg = self.channel_bg[1]
        
        start_index = df.first_valid_index()
        end_index = df.last_valid_index()
        

        for ax in axes:
            ax.set_facecolor(face_color)
            ax.tick_params(axis='x', labelsize=self.fontsize, colors=tick_color)
            ax.tick_params(axis='y', labelsize=self.fontsize, colors=tick_color)

        x = 'human open time'
        
        
        # ---------------------------------------------------------------------
        # first subplot
        lw = 50 / (len(df)/10)
        
        # plot vertical lines for candles with active position
        if 'p.actv' in df.columns:
            for x_ in range(len(df)):
                if df.iloc[x_]['p.actv'] == True:
                    ax1.axvline(x=x_, color=color_position, alpha=alpha_position, 
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
            df.plot(kind='line', x=x, y=columns[0], color=line4, linewidth=0.1, \
                    alpha=alpha_line4, ax=ax1)
            df.plot(kind='line', x=x, y=columns[1], color=line3, linewidth=0.1, \
                    alpha=alpha_line3, ax=ax1)
        except Exception as e:
            print(e)
            sys.exit()

        if 'ewm.63' in df.columns:
            df.plot(kind='line', x=x, y='ewm.63', color=line3, linewidth=0.5, \
                    linestyle='dotted', alpha=alpha_line3, ax=ax1)

        # plot STOP LOSS levels
        if 'sl.current' in df.columns:
            df.plot(kind='line', x=x, y='sl.current', color=line1, 
                    linewidth=0.1, alpha=0.8, drawstyle="steps-mid", ax=ax1)

        # plot KELTNER CHANNEL
        if 'kc.upper' in df.columns:
            df.plot(kind='line', x=x, y='kc.upper', color=channel, 
                    linestyle='dotted', linewidth=0.1, ax=ax1)

            df.plot(kind='line', x=x, y='kc.lower', color=channel, 
                    linestyle='dotted', linewidth=0.1, ax=ax1)

            df.plot(kind='line', x=x, y='kc.mid', color=channel, 
                    linestyle='-', linewidth=0.2, ax=ax1)

            ax1.fill_between(x=df['human open time'], y1=df['kc.upper'], 
                             y2=df['kc.lower'], color=channel_bg, 
                             edgecolor=channel, alpha=alpha_channel_bg,
                             linewidth=0.1, zorder=-5)

        # ---------------------------------------------------------------------
        # plot all the CANDLES - color coded for market state
        width = round(150 / (len(df)), 4)
        width2 = round(width / 4, 4)
        # width = 0.4
        # width2 = 0.05
        alpha_bull = 1
        
        for idx in range(start_index+1, end_index+1):
            x_ = idx - start_index
            _open, _high = df['open'].iloc[x_], df['high'].iloc[x_]
            _low, _close = df['low'].iloc[x_], df['close'].iloc[x_]
            state = df['s.state'].iloc[x_]
            
            if state == 'bull':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=bull, alpha=alpha_bull)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bull, alpha=alpha_bull)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bull, alpha=alpha_bull)  

                else:       
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bull, alpha=alpha_bull)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bull, alpha=alpha_bull)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bull, alpha=alpha_bull)
                    
            elif state == 'bear':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=bear, alpha=alpha_bear)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bear, alpha=alpha_bear)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bear, alpha=alpha_bear)  

                else:            
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bear, alpha=alpha_bear)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bear, alpha=alpha_bear)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bear, alpha=alpha_bear)

            elif state == 'flat':
            
                if _close > _open:
                    ax1.bar(x_, height=_close - _open, width=width, bottom=_open, color=flat, alpha=alpha_flat)
                    ax1.bar(x_, height=_high - _close, width=width2, bottom=_close, color=flat, alpha=alpha_flat)
                    ax1.bar(x_, height=_open - _low, width=width2, bottom=_low, color=flat, alpha=alpha_flat)  

                else:            
                    ax1.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=flat, alpha=alpha_flat)
                    ax1.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=flat, alpha=alpha_flat)
                    ax1.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=flat, alpha=alpha_flat)            


        # ---------------------------------------------------------------------
        df.plot(x=x, y='buy.price', alpha=alpha_buy, marker='^', color=color_buy, 
                markersize=2, ax=ax1)
        df.plot(x=x, y='sell.price', alpha=alpha_sell, marker='v',color=color_sell, 
                markersize=2, ax=ax1)
        

        # ---------------------------------------------------------------------
        # second subplot
        if 'stoch.close' in df.columns:
            ax2.set_ylim(0,100)
            
            if 's.stc' in df.columns:
                
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.stc'] == 1: 
                        ax2.axvline(x_, color=color_buy, linewidth=0.2, alpha=alpha_buy)
                    if df.loc[idx, 's.stc'] == -1: 
                        ax2.axvline(x_, color=color_sell, linewidth=0.2, alpha=alpha_sell)
                    
            df.plot(x=x, y='stoch.close.k',kind='line', color=line1, alpha=alpha_line1, 
                    linewidth=0.1, ax=ax2)
            df.plot(x=x, y='stoch.close.d',kind='line', color=line2, alpha=alpha_line2, 
                    linewidth=0.1, ax=ax2)
            
                    
        # ---------------------------------------------------------------------
        # third subplot
        if 'stoch.rsi.k' in df.columns:
            ax3.set_ylim(0,100)
            
            if 's.stc.rsi' in df.columns:
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.stc.rsi'] == 1: 
                        ax3.axvline(x_, color=color_buy, linewidth=0.2, alpha=alpha_buy)
                    if df.loc[idx, 's.stc.rsi'] == -1:
                        ax3.axvline(x_, color=color_sell, linewidth=0.2, alpha=alpha_sell)

            
            df.plot(kind='line', x=x, y='stoch.rsi.k', color=line1 , alpha=alpha_line1, linewidth=0.1, ax=ax3)
            df.plot(kind='line', x=x, y='stoch.rsi.d', color=line2 , alpha=alpha_line2, linewidth=0.1, ax=ax3)
            ax3.axhline(y = 90, color = tick_color, linestyle = 'dotted', linewidth=0.5)
            ax3.axhline(y = 10, color = tick_color, linestyle = 'dotted', linewidth=0.5)            

        else:
            ax3.set_ylim(0,1)
            df.plot(kind='line', x=x, y='cptl.drawdown', color=line1 , alpha=alpha_line1, linewidth=0.2, ax=ax3)
            df.plot(kind='line', x=x, y='hodl.drawdown', color=line2 , alpha=alpha_line2, linewidth=0.1, ax=ax3)

        # ---------------------------------------------------------------------
        # fourth subplot
        if 'mom' in df.columns:
            columns = [col for col in df.columns if 'mom.' in col]
            ax4.set_ylim(-10.5, 10.5)

            try:
                df.plot(x=x, y=columns[0], kind='line', color='red', alpha=0.8, 
                        linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)

            try:
                df.plot(x=x, y=columns[1], kind='line', color=line1, 
                        alpha=alpha_line1, linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)

            try:
                df.plot(x=x, y=columns[2], kind='line', color=line2, 
                        alpha=alpha_line2, linewidth=0.1, ax=ax4)
            except Exception as e: 
                print(e)
            
            if 's.mom' in df.columns:
                for idx in df.index:
                    x_ = idx - start_index
                    if df.loc[idx, 's.mom'] == 1: ax4.axvline(x_, color=color_buy, alpha=alpha_buy, linewidth=0.2)
                    if df.loc[idx, 's.mom'] == -1: ax4.axvline(x_, color=color_sell, alpha=alpha_sell, linewidth=0.2)
                    
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
            df.plot(kind='line', x=x, y='ATR percent', color=line2, alpha=0.3, linewidth=0.5, ax=ax4)

        # ---------------------------------------------------------------------
        # fifth subplot
        if 's.cci' in df.columns:
            for idx in df.index:
                x_ = idx - start_index
                if df.loc[idx, 's.cci'] == 1: 
                    ax5.axvline(x_, color=color_buy, linewidth=0.2, alpha=alpha_buy)
                if df.loc[idx, 's.cci'] == -1:
                    ax5.axvline(x_, color=color_sell, linewidth=0.2, alpha=alpha_sell)

        df.plot(x=x, y='cci', kind='line', color=line1, alpha=alpha_line1, 
                linestyle='solid', linewidth=0.2, ax=ax5)        
        
        
        
        # ---------------------------------------------------------------------
        # sixth subplot
        df.plot(kind='line', x=x, y='capital', color=line4, alpha=alpha_line4, 
                linewidth=0.5, ax=ax6)
        
        if 'hodl.value' in df.columns:
            # ax7 = ax6.twinx()
            hodl_color = tick_color
            
            df.plot(kind='line', x=x, y='hodl.value', color=line3, alpha=alpha_line3, 
                    linestyle='dotted', linewidth=0.3, ax=ax6)
            
            ax6.tick_params(axis ='y', labelcolor=hodl_color, labelsize=self.fontsize)


        # ---------------------------------------------------------------------
        for ax_ in axes:
            ax_.grid(which='both', linewidth=0.1, linestyle='dotted', 
                     color=grid, alpha=alpha_grid)
            
            ax_.legend(fontsize=self.fontsize)
            ax_.tick_params(axis='both', which='both', color=tick_color)
            
            # ax_.set_frame_on(False)            
            # ax_.xaxis_date(tz=None)
            
            for brdr in ['left', 'right', 'top', 'bottom']:
                ax_.spines[brdr].set_color(color_position) 
                ax_.spines[brdr].set_linewidth(0.3)

        # ax7.legend(fontsize=self.fontsize)
        # ax7.legend().set_visible(False)

        # title = f'{PAIR} ({INTERVAL}) from {START} to {END}'
        # plt.title(title, y=1.0)
        
        # this controls the number/spacing of ticks on the plot
        # number_of_major_ticks = 20
        # number_of_minor_ticks = 5
        # ax1.xaxis.set_major_locator(MultipleLocator(number_of_major_ticks))
        # ax1.xaxis.set_minor_locator(MultipleLocator(number_of_minor_ticks))

        ax1.yaxis.set_major_locator(MultipleLocator(df['close'].max()/10))
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        
        # fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()        

        return



# ============================================================================= 
class AnalystChart(Minerva):
    
    def __init__(self, df:pd.DataFrame, subplots:dict, color_scheme:str):
        
        Minerva.__init__(self, color_scheme=color_scheme)
        
        self.df = self._prepare_dataframe(df)
        
        self.subplots = subplots
        self.counter = 1
        self.no_of_subplots = len(subplots) + 1
        
        self.fontsize = 4
        self.x = 'human open time'


    # -------------------------------------------------------------------------
    def draw(self):
        
        figsize=(12, 5.2)
        
        if self.no_of_subplots > 1:        
            ratios = [3]
            for _ in range(self.no_of_subplots-1): 
                ratios.append(1)

            self.fig, self.axes = plt.subplots(self.no_of_subplots, 1,
                                                figsize=figsize,
                                                sharex=True,
                                                sharey=False,
                                                gridspec_kw={'width_ratios' : [1],
                                                                'height_ratios' : ratios})
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
        
        self.fig.set_dpi(150)
        self.fig.patch.set_facecolor(self.canvas)
        
        # ---------------------------------------------------------------------
        # prepare subplotes
        self._moving_averages()
        self._ohlcv()
        self._buys_and_sells()
        
        for indicator_name in self.subplots.keys():
            self._indicator(indicator_name)

        
        # ---------------------------------------------------------------------
        # different format parameters
        _lst = self.axes if self.no_of_subplots > 1 else [self.axes]
        
        for ax_ in _lst:
            
            ax_.set_facecolor(self.background)
            ax_.tick_params(axis='x', labelsize=self.fontsize, colors=self.tick[0])
            ax_.tick_params(axis='y', labelsize=self.fontsize, colors=self.tick[0])
            
            ax_.grid(which='both', linewidth=0.075, linestyle='dotted', 
                        color=self.grid[0], alpha=self.grid[1])
            
            ax_.legend(fontsize=self.fontsize)
            
            ax_.margins(tight=True)
            
            ax_.legend(fancybox=True, framealpha=0.5, shadow=False,\
                borderpad=1, labelcolor=self.grid[0], facecolor=self.canvas,\
                fontsize=self.fontsize, edgecolor=self.canvas)
            
            for brdr in ['left', 'right', 'top', 'bottom']:
                ax_.spines[brdr].set_color(self.position[0]) 
                ax_.spines[brdr].set_linewidth(0.3)
                
        # ---------------------------------------------------------------------
        # plot the whole thing
        plt.tight_layout()
        plt.show()
        
    # -------------------------------------------------------------------------
    def _positions(self):
        
        raise NotImplementedError
    
    def _channel(self, axes_, upper_col, lower_col, mid_col):
        
        axes_.fill_between(x=self.df.index, 
                           y1=self.df['kc.upper'], y2=self.df['kc.lower'], 
                           color=self.channel_bg[0], edgecolor=self.channel[0], 
                           alpha=self.channel_bg[1], linewidth=0.1, zorder=-5)
    
    
    def _moving_averages(self):
        try:
            ax = self.axes[0]
        except:
            ax = self.axes
            
        colors = [self.line1[0], self.line2[0], self.line3[0], self.line4[0]]
        alpha = [self.line1[1], self.line2[1], self.line3[1], self.line4[1]]
        
        columns = [col for col in self.df.columns \
            if col.split('.')[0] == 'sma' or col.split('.')[0] == 'ewm']
        
        for idx, col in enumerate(columns):
            lw = round((idx+1)/3, 3)
            label = col.split('.')[0].upper() + ' ' + col.split('.')[1]
            ax.plot(self.df[col], color=colors[idx], linewidth=lw, \
                    alpha=alpha[idx], label=label)            
    
    def _ohlcv(self):
        '''Plot all the OHLCV candles - color coded for market state'''
        
        def _get_ax_size(ax):
            bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= self.fig.dpi
            height *= self.fig.dpi
            return width, height
        
        def _get_linewidth() -> tuple:
            
            df = self.df
            values = len(self.df)
            precision = 5
            
            open_time = df.index.astype(int)
            timedelta = int((open_time[1] - open_time[0]) / 1_000_000_000)
            print(f'{timedelta=}')
                        
            _, ax_width = _get_ax_size(ax)
            corr_factor = 86400 / timedelta
            value_factor = values / 1000 

            width = round(
                ((ax_width * 1.5) / (values) * value_factor) / corr_factor, 
                precision
                )

            width2 = round(width / 3, precision)            
            return width, width2
        
        try:
            ax = self.axes[0]
            ax.set_title(self.title, y=1.0, pad=-14)
        except:
            ax = self.axes

        # if (self.df['high'].max() / self.df['low'].min()) > 10:
        #     ax.set_yscale('log') 
        #     ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        #     ax.yaxis.set_minor_formatter(NullFormatter())
        
        width, width2 = _get_linewidth()        
        print(f'{len(self.df)=} :: {width=}, {width2=}')

        # plot candles with color depending on market regime 
        for x_, row in self.df.iterrows():
            _open, _high = row['open'], row['high']
            _low, _close = row['low'], row['close']
            state = row['s.state']
            
            if state == 'bull':          
                if _close > _open:
                    ax.bar(x_, height=_close - _open, width=width, bottom=_open, color=bull, alpha=alpha_bull)
                    ax.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bull, alpha=alpha_bull)
                    ax.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bull, alpha=alpha_bull)  
                else:            
                    ax.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bull, alpha=alpha_bull)
                    ax.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bull, alpha=alpha_bull)
                    ax.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bull, alpha=alpha_bull)                
            
            elif state == 'bear':
            
                if _close > _open:
                    ax.bar(x_, height=_close - _open, width=width, bottom=_open, color=bear, alpha=alpha_bear)
                    ax.bar(x_, height=_high - _close, width=width2, bottom=_close, color=bear, alpha=alpha_bear)
                    ax.bar(x_, height=_open - _low, width=width2, bottom=_low, color=bear, alpha=alpha_bear)  
                else:            
                    ax.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=bear, alpha=alpha_bear)
                    ax.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=bear, alpha=alpha_bear)
                    ax.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=bear, alpha=alpha_bear)
            
            elif state == 'flat':
            
                if _close > _open:
                    ax.bar(x_, height=_close - _open, width=width, bottom=_open, color=flat, alpha=alpha_flat)
                    ax.bar(x_, height=_high - _close, width=width2, bottom=_close, color=flat, alpha=alpha_flat)
                    ax.bar(x_, height=_open - _low, width=width2, bottom=_low, color=flat, alpha=alpha_flat)  

                else:            
                    ax.bar(x=x_, height=_open - _close, width=width, bottom=_close, color=flat, alpha=alpha_flat)
                    ax.bar(x=x_, height=_high - _open, width=width2, bottom=_open, color=flat, alpha=alpha_flat)
                    ax.bar(x=x_, height=_close - _low, width=width2, bottom=_low ,color=flat, alpha=alpha_flat)         
           
    def _buys_and_sells(self):
        '''Plot markers for buy/sell signals and add the columns if missing.'''
        try:
            ax = self.axes[0]
        except:
            ax = self.axes
            
        markersize = 20
        
        if not 'buy.price' in self.df.columns:
            self.df['buy.price'] = np.nan
            self.df.loc[self.df['s.all'].shift() > 0, 'buy.price'] \
                = self.df['open']
               
        ax.scatter(x=self.df.index, y=self.df['buy.price'], 
                   alpha=self.buy[1], marker='^', color=self.buy[0], 
                   s=markersize)
            
        if not 'sell.price' in self.df.columns:
            self.df['sell.price'] = np.nan
            self.df.loc[self.df['s.all'].shift() < 0, 'sell.price'] \
                = self.df['open'] 
            
        ax.scatter(x=self.df.index ,y=self.df['sell.price'], 
                   alpha=self.sell[1], marker='v', color=self.sell[0], 
                   s=markersize)        
                  
    # -------------------------------------------------------------------------
    def _indicator(self, indicator_name:str):
        
        params = self.subplots[indicator_name]   
        ax = self.axes[self.counter]

        for idx, col in enumerate(params['columns']):
            ax.plot(self.df[col], color=self.line_colors[idx], \
                    linewidth=0.5, alpha=self.line_alphas[idx],
                    label=indicator_name)
            
        for idx, line in enumerate(params['horizontal lines']):
            ax.axhline(y=line, color=self.hline[0], \
                linestyle='dotted', linewidth=0.5)
            
        if params['fill']:
            if params['horizontal lines']:
                ax.fill_between(x=self.df.index,
                                y1=params['horizontal lines'][0], 
                                y2=params['horizontal lines'][1],
                                color=self.channel_bg[0],
                                alpha=self.channel_bg[1]
                                )
        try:    
            if params['channel']:
                col_upper = params['channel'][0]
                col_lower = params['channel'][1]

                ax.fill_between(x=self.df.index,
                                y1=self.df[col_upper], 
                                y2=self.df[col_lower],
                                color=self.channel_bg[0],
                                alpha=self.channel_bg[1]
                                )
        except:
            pass
        
        if params.get('signal') is not None:
            sig_col = params['signal']
            
            for idx in self.df.index:
                x_ = idx
                
                if self.df.loc[idx, sig_col] == 1: 
                    ax.axvline(x_, color=self.buy[0], linewidth=0.2, 
                                alpha=self.buy[1])
                
                if self.df.loc[idx, sig_col] == -1:
                    ax.axvline(x_, color=self.sell[0], linewidth=0.2, 
                                alpha=self.sell[1])
            
        self.counter += 1
            
        
        
    # -------------------------------------------------------------------------
    def _prepare_dataframe(self, df:pd.DataFrame) -> pd.DataFrame:
        
        df.dropna(inplace=True)
        df = self._add_trend_signal(df=df)
        df['human open time'] = pd.to_datetime(df['human open time'])
        df = df.set_index('human open time')
        
        return df




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
        # df.set_index('human open time', drop=True, inplace=True)
        df.dropna(inplace=True)
        for col in slope_cols: df[col] = df[col].astype(float)
        for col in signal_cols: df[col] = df[col].astype(int)
        
        # if not 'kc.upper' in df.columns:
        #     i = Indicators()
        #     df = i.keltner_channel(df=df)


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


        x = 'human open time'

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

        df.plot(kind='line', y='close', color='fuchsia', alpha=alpha_high, linewidth=thicker, ax=ax_all)
        df.plot(kind='line', y=SHORT_SMA, color='blue', linewidth=thinner, alpha=alpha_high, ax=ax_all)
        df.plot(kind='line', y=LONG_SMA, color='grey', linewidth=thinner, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='kc.upper', color='grey', 
                linestyle='dotted', linewidth=0.2, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='kc.lower', color='grey', 
                linestyle='dotted', linewidth=0.2, alpha=alpha_high, ax=ax_all)

        df.plot(kind='line', y='close', color='fuchsia', alpha=alpha_high, linewidth=thicker, ax=ax_trend)
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

        #df.set_index('human open time', drop=True, inplace=True)
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
        x = 'human open time'
        df.plot.scatter(x=x, y='low', s=[0.1], c='red', alpha=0.5, ax=ax_atr_sl)
        df.plot(kind='line', x=x, y='open', color='blue', linewidth=0.2, drawstyle="steps-mid", ax=ax_atr_sl)
        df.plot(kind='line', x=x, y='sl.atr', color='fuchsia', linewidth=0.3, drawstyle="steps-mid", ax=ax_atr_sl)
        # df.plot(kind='line', y='low', color='darkslategray', alpha=0.5, linestyle='dotted', linewidth=0.5, ax=ax_atr_sl)

        # ------------------------------------------------------------------------- 
        # plot Open + Low price and PERCENTILE stop-loss in SECOND subplot
        
        df.plot(kind='line', x=x, y='low', color='blue', alpha=0.7, linewidth=0.2, ax=ax_pctl_sl)
        df.plot(kind='line', x=x, y='sl.pctl', color='fuchsia', linewidth=0.2, drawstyle="steps-mid", ax=ax_pctl_sl)
        df.plot(kind='line', x=x, y='open', color='darkslategray', alpha=0.5, linestyle='dotted', linewidth=0.5, ax=ax_pctl_sl)

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

        fig = go.Figure(data=[go.Candlestick(x=self.df['human open time'],
                                             open=self.df['open'],
                                             high=self.df['high'],
                                             low=self.df['low'],
                                             close=self.df['close']
                                             ),
                              go.Scatter(x=self.df['human open time'], 
                                          y=self.df['sl.long'],
                                          visible=True,
                                          name='SL LONG',
                                          line = {"shape": 'hvh',
                                                  "color" : 'azure'}),
                              go.Scatter(x=self.df['human open time'], 
                                          y=self.df['sl.short'],
                                          visible=True,
                                          name='SL SHORT ATR',
                                          line = {"shape": 'hvh',
                                                  "color" : 'azure'}),
                            #   go.Scatter(x=self.df['human open time'], 
                            #               y=self.df['sl.l.pctl'],
                            #               name='SL LONG PCTL',
                            #               visible=True,
                            #               line = {"shape": 'hvh',
                            #                       "color" : 'crimson'},
                            #                       hovertext=self.df['sl.s.pctl.pct']),
                            #   go.Scatter(x=self.df['human open time'], 
                            #               y=self.df['sl.s.pctl'],
                            #               visible=True,
                            #               name='SL SHORT PCTL',
                            #               line = {"shape": 'hvh',
                            #                       "color" : 'crimson'},
                            #               hovertext=self.df['sl.s.pctl.pct']),
                            #   go.Scatter(x=self.df['human open time'], 
                            #               y=self.df['tp.l.atr'],
                            #               name='TP LONG ATR',
                            #               visible=True,
                            #               line = {"shape": 'hvh',
                            #                       "color" : 'deeppink'}),
                            #   go.Scatter(x=self.df['human open time'], 
                            #               y=self.df['tp.s.atr'],
                            #               name='TP SHORT ATR',
                            #               visible=True,
                            #               line = {"shape": 'hvh',
                            #                       "color" : 'deeppink'}),
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

    pass