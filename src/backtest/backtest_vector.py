#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Union
from string import ascii_lowercase, digits
from random import choice

# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------
from models.symbol import Symbol
from staff.moses import Moses
from staff.moneta import Moneta
from staff.shakti import Shakti

# =============================================================================
class Backtest:
    '''This class runs a single backtest (1 symbol, 1 interval)'''

    def __init__(self, symbol: Symbol, interval: str, start: Union[int,str], 
                 end: Union[int,str], leverage: float, risk_level:int, 
                 strategy: str, state_params:dict, 
                 stop_loss_strategy: Union[str, None], 
                 stoploss_params: Union[dict, None],
                 verbose:bool=False, draw_chart:bool=True):

        self._ex_time_start: float = time.time()
        self._ex_time_end: float = 0
        self.execution_time: float = 0
 
        self.symbol: Symbol = symbol
        self.interval: str = interval
        self.start: Union[int,str] = start
        self.end: Union[int,str] = end
        
        self.INITIAL_CAPITAL: float = state_params.get('initial_capital', 0)
        self.MAX_LEVERAGE: float = leverage
        self.RISK_LEVEL: int = risk_level
        
        self.df: pd.DataFrame
        
        self.state_params: dict = state_params
        self.stop_loss_strategy: Union[str, None] = stop_loss_strategy
        self.stoploss_params: Union[dict, None] = stoploss_params

        self.fees = 0.00
        self.slippage = 0.00
        self.trade_costs = self.fees + self.slippage

        if stop_loss_strategy is not None:
            self.moses = Moses()
            self.moses.set_sl_strategy(stop_loss_strategy, stoploss_params)

        self.moneta = Moneta(
            symbol=self.symbol, interval=self.interval, strategy=strategy,
            start=start, end=end, initial_capital=self.INITIAL_CAPITAL,
            risk_level=self.RISK_LEVEL
        )

        self.shakti = Shakti()

        # initialize yourself
        self.initialize()

    # -------------------------------------------------------------------------
    def initialize(self):

        self.df = self.moneta.data
        
        if self.stop_loss_strategy:
            self.df = self.moses.get_stop_loss_prices(self.moneta.data)
            self.df['sl.long'] = self.df['sl.long'].shift()
            self.df['sl.l.trig'] = False
            self.df['sl.short'] = self.df['sl.short'].shift()
            self.df['sl.s.trig'] = False
        
        for col in self.df.columns:
            if 's.' in col:
                self.df[col] = self.df[col].shift()
                
        self.df = self.__delete_unnecessary_columns(self.df)

    def run(self):
        
        self.df['returns.pct'] = self.df['close'].pct_change() 
        
        df = self.df.loc[200:, :].copy(deep=True)
        
        df = self.find_positions(df)
        
        self.show_overview(df)
        sys.exit()
        
        df = self.add_leverage(df)
        
        df = self.calculate_trades(df)
        df = self.add_returns(df)
                
        df = self.get_stop_loss_percent(df)
        df = self.calculate_result_statistics(df)
                        
        df['sl.current'].replace(0, np.nan, inplace=True)

        df = self.find_buy_and_sell_amounts(df)
    
        # self.show_overview(df)
        # sys.exit()
                
        return df

        # ......................................................................
        
    def find_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['event.id'] = np.random.rand(len(df))
        df['event.id'] = (df['event.id'] * 10000).astype(int)

        df['position'] = df['s.all']
        position = df['position']
        position.replace(0.0, np.nan, inplace=True)
        position.ffill(inplace=True)
                
        df.loc[df['s.all'] == 1, 'p.type'] = 'LONG'
        df.loc[df['s.all'] == -1, 'p.type'] = 'SHORT'

        df.loc[(df['s.all'] == 0), 'event.id'] = np.nan
        df.loc[(df['position'] == df['position'].shift()), 'event.id'] = np.nan
                
        df['event.id'].ffill(inplace=True) 
        
        return df

        df = self._apply_trailing_stop_loss(df)
        df = self.add_stop_loss(df)
        df = self.add_sl_triggered(df)
                
        df = self.add_position_change(df)
        df = self.add_position_active(df)
        
        df = self.first_cleanup(df)
        df = self.reevaluate_position_change(df)
        
        df = self.find_buys_and_sells(df)
        df = self.find_buy_and_sell_prices(df)
        
        df = self.add_leverage(df)
        
        return df

    def _apply_trailing_stop_loss(self, df) -> pd.DataFrame:
        
        if self.stop_loss_strategy is not None:    
            df['sl.short'] = df.groupby('event.id')['sl.short'].cummin()
            df['sl.long'] = df.groupby('event.id')['sl.long'].cummax()
            
            df.loc[df['sl.short'] < df['high'], 'sl.s.trig'] = True
            df.loc[df['sl.long'] > df['low'], 'sl.l.trig'] = True
   
        else:
            df['sl.l.trig'] = False
            df['sl.s.trig'] = False        
        return df
          
    def add_stop_loss(self, df) -> pd.DataFrame:
        if self.stop_loss_strategy is not None: 
            df.loc[(df['position'] == 1), 'sl.current'] = df['sl.long']
            df.loc[(df['position'] == -1), 'sl.current'] = df['sl.short']
        else:
            df['sl.current'] = np.nan
               
        return df
    
    def get_stop_loss_percent(self, df) -> pd.Series:
        df['sl.pct'] = (df['sl.current'] / df['open'])
        df.loc[df['sl.pct'] != 0, 'sl.pct'] = (df['sl.pct'] -1).round(4)
        return df

    def add_position_change(self, df) -> pd.DataFrame:
        
        conditions = [(df['s.all'] == 1),
                      (df['s.all'] == -1),
                      (df['sl.trig'] == True),
        ]
        
        choices = [1, 1, -1]
        df['p.change'] = np.select(conditions, choices, 0)
        
        return df        

    def add_sl_triggered(self, df) -> pd.DataFrame:
        if self.stop_loss_strategy is not None: 
            df['_tmp'] = df['p.type']
            df['_tmp'].ffill(inplace=True)
            
            conditions = [
                (df['_tmp'] == 'LONG') & (df['sl.l.trig'] == True),
                (df['_tmp'] == 'SHORT') & (df['sl.s.trig'] == True),
            ]
            choices = [True, True]
            df['sl.trig'] = np.select(conditions, choices, False)
            df.drop('_tmp', axis=1, inplace=True)
        else:
            df['sl.trig'] = False
                
        return df
     
    def add_position_active(self, df) -> pd.DataFrame:
        conditions = [
            ~(df['s.all'] == 0),
            (df['sl.trig'].shift() == True),
        ]
        choices = ['•', '-']
        df['p.actv'] = np.select(conditions, choices, None)
        df['p.actv'].ffill(inplace=True)
        return df
     
    def first_cleanup(self, df) -> pd.DataFrame:
        
        df.loc[(df['p.actv'] == '-'), 'p.change'] = 0
        df.loc[(df['p.actv'] == '-'), 'position'] = 0
        df.loc[(df['p.actv'] == '-'), 'sl.current'] = np.nan
        df.loc[(df['p.actv'] == '-'), 'sl.trig'] = False
        
        df.loc[df['p.actv'] == '-', 'p.type'] = ''
        df['p.type'].ffill(inplace=True)
        
        df['p.actv'].replace('-', '', inplace=True)
        
        return df
    
    def reevaluate_position_change(self, df) -> pd.DataFrame:

        df.loc[(df['p.type'] == df['p.type'].shift())
            & (df['sl.trig'].shift() == False)
            & ~(df['s.all'] == 0), 
            'p.change' 
            ] = 0
        
        df.loc[
            (df['p.type'] == 'LONG') & (df['p.type'].shift() == 'SHORT')
            | (df['p.type'] == 'SHORT') & (df['p.type'].shift() == 'LONG')
            & (df['sl.trig'].shift() == False),
            'p.change'
        ] = 2
        
        df.loc[
            (df['p.type'] == 'LONG') & (df['sl.trig'] == True),
            'p.change' 
        ] = -1
        
        df.loc[
            (df['p.type'] == 'SHORT') & (df['sl.trig'] == True),
            'p.change' 
        ] = -1
        
        return df
    
    def find_buys_and_sells(self, df) -> pd.DataFrame:

        conditions = [
            (df['p.type'] == 'LONG') & (df['p.change'] > 0),
            (df['p.type'] == 'SHORT') & (df['p.change'] < 0),
        ]
        choices = ['•', '•']
        df['buy'] = np.select(conditions, choices, '')

        conditions = [
            (df['p.type'] == 'LONG') & (df['p.change'] < 0),
            (df['p.type'] == 'SHORT') & (df['p.change'] > 0),
        ]
        df['sell'] = np.select(conditions, choices, '')
        
        # if at the end of the test period we have an open position, close it!
        # if df.loc[df.last_valid_index(), 'p.type'] == 'LONG':
        #     df.loc[df.last_valid_index(),'sell'] = '•'
        # elif df.loc[df.last_valid_index(), 'p.type'] == 'SHORT':
        #     df.loc[df.last_valid_index(), 'buy'] = '•'
            
        # if df.loc[df.last_valid_index(), 'p.type'] != '':
        #     df.loc[df.last_valid_index(),'p.change'] = -1

       
        
        return df
    
    def find_buy_and_sell_prices(self, df) -> pd.DataFrame:

        df.loc[
            (df['buy'] == '•'), # & (df['s.all'] == 1), 
            'buy.price'
            ] = df['open']

        df.loc[
            (df['sell'] == '•') & (df['s.all'] == -1), 
            'sell.price'
            ] = df['open']
        
        # set buy/sell price for SL triggered positions
        df.loc[
            (df['p.type'] == 'LONG') & (df['sl.trig']), 
            'sell.price'
            ] = df['sl.current']

        df.loc[
            (df['p.type']== 'SHORT') & (df['sl.trig']), 
            'buy.price'
            ] = df['sl.current']
        
        return df

    def find_buy_and_sell_amounts(self, df) -> pd.DataFrame:
        
        # position opened  
        df.loc[
            (df['buy'] == '•') & (df['position'].shift() != 1), 
            'buy.amount'
            ] = (df['b.value'] / df['close']) * df['p.size']
        
        df.loc[
            (df['sell'] == '•') & (df['position'].shift() != -1), 
            'sell.amount'
            ] = (df['b.value'] / df['close']) * df['p.size']
        
                
        df['p.size'] = np.nan
        
        df.loc[df['buy.amount'] != 0, 'p.size'] = df['buy.amount']
        
        df.loc[
            (df['sell'] == '•') & (df['position'].shift() == 0),  
            'p.size'
            ] = df['sell.amount']
        
        df['p.size'].ffill(inplace=True)
        
        # position closed
        df.loc[
            (df['sell'] == '•') & (df['position'] == 1), 
            'sell.amount'
            ] = df['p.size']

        df.loc[
            (df['buy'] == '•') & (df['position'] == -1), 
            'buy.amount'
            ] = df['p.size']
        
        df.loc[~(df['p.actv'] == '•'), 'p.size'] = 0
        
        return df

    
    # .........................................................................
    def _get_max_position_size(self, row:pd.Series) -> float:
        
        return 1

    def add_leverage(self, df:pd.DataFrame) -> pd.DataFrame:
        
        df.loc[
            (df['p.type'] == 'LONG') & ~(df['p.type'].shift() == 'LONG'),
            'leverage'   
            ] =  df['leverage.max']

        df.loc[
            (df['p.type'] == 'SHORT') & ~(df['p.type'].shift() == 'SHORT'),
            'leverage'   
            ] = df['leverage.max']
        
        df.loc[df['p.actv'] == '', 'leverage'] = 0
        
        df['leverage'].clip(upper=self.MAX_LEVERAGE, inplace=True)
        
        
        df['leverage'].ffill(inplace=True)

        return df

    def add_returns(self, df):

        # my_returns = df['p.size'] * df['returns.log'] * df['position']
        # df['s.return'] = my_returns.cumsum().apply(np.exp)  
        
        df['returns.log'] = df['close'].apply(np.log).diff(1)
        
        df['s.returns'] = 0
        df.loc[
            df['p.actv'] == '•', 
            's.returns'
            ] = df['returns.pct'] * df['p.size'] * df['position']
        
        df.loc[
            (df['sell'] == '•') & (df['sl.trig'] == True),
            's.returns'
            ] = (df['sl.current'] / df['close'].shift() - 1) * df['p.size']
        
        df.loc[
            (df['buy'] == '•') & (df['sl.trig'] == True),
            's.returns'
            ] = (df['sl.current'] / df['close'].shift() - 1) * df['p.size'] * -1
         
        return df           
    

    # --------------------------------------------------------------------------
    def calculate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df.at[df.first_valid_index(), 'b.quote'] = self.INITIAL_CAPITAL
        
        for idx in range(df.first_valid_index() + 1, df.last_valid_index() + 1):
            
            if df.at[idx, 'buy'] == '•':
                df = self.__process_buy(df, idx)
              
                # this covers the case where a LONG SL was triggered 
                # during the same interval where we bought
                if df.at[idx, 'sl.trig'] and df.at[idx, 'p.type'] == 'LONG':
                    df.at[idx, 'b.quote'] = df.at[idx, 'b.base'] * df.at[idx, 'sl.current']
                    df.at[idx, 'b.base'] = 0
   
                
            elif df.at[idx, 'sell']:
                
                df = self.__process_sell(df, idx)
                
                # this covers the case where a SHORT SL was triggered 
                # during the same interval where we bought
                if df.at[idx, 'sl.trig'] and df.at[idx, 'p.type'] == 'SHORT':
                    df.at[idx, 'b.quote'] = df.at[idx, 'b.quote'] \
                        + df.at[idx, 'b.base'] * df.at[idx, 'sl.current']
                    df.at[idx, 'b.base'] = 0
                 
            else:
                df.at[idx, 'b.base'] = df.at[idx-1, 'b.base']
                df.at[idx, 'b.quote'] = df.at[idx-1, 'b.quote']
                
        df['b.value'] = df['b.quote'] + (df['b.base'] * df['close']) 
        
        return df

    def __process_buy(self, df, idx):
        
        # openeing a LONG position
        if df.at[idx, 'p.type'] == 'LONG':
            
            # print(idx, ': processing buy ...')
            
            # if we were in a SHORT position immediately before, close it first
            if df.at[idx-1, 'b.base'] < 0:
                df.at[idx, 'b.quote'] = df.at[idx-1, 'b.quote'] \
                    - (-1 * df.at[idx-1, 'b.base'] * df.at[idx, 'buy.price'])
            else:
                 df.at[idx, 'b.quote'] =  df.at[idx-1, 'b.quote']
            
            df.at[idx, 'b.base'] = \
                ((df.at[idx, 'b.quote'] * df.at[idx, 'leverage']) \
                    / df.at[idx, 'buy.price']) \
                    * (1 - self.trade_costs)
                
            df.at[idx, 'b.quote'] = df.at[idx, 'b.quote'] \
                - df.at[idx, 'b.base'] * df.at[idx, 'buy.price']
        
        # closing a SHORT position
        elif df.at[idx, 'p.type'] == 'SHORT':

            df.at[idx, 'b.quote'] = df.at[idx-1, 'b.quote'] \
                + df.at[idx-1, 'b.base'] * df.at[idx, 'buy.price']
            
            df.at[idx, 'b.base'] = 0
        
        return df
    
    def __process_sell(self, df, idx):
        
        # closing a LONG position
        if df.at[idx, 'p.type'] == 'LONG':

            df.at[idx, 'b.quote'] = df.at[idx-1, 'b.quote'] \
                + df.at[idx-1, 'b.base'] * df.at[idx, 'sell.price']
            df.at[idx, 'b.base'] = 0

        # opening a  SHORT position
        elif df.at[idx, 'p.type'] == 'SHORT':

            # if we were in a LONG position immediately before, close it first
            if df.at[idx-1, 'p.type'] == 'LONG':
                budget = df.at[idx-1, 'b.quote'] \
                    + (df.at[idx-1, 'b.base'] * df.at[idx, 'sell.price'])    
            else:
                budget = df.at[idx-1, 'b.quote']
                

            df.at[idx, 'b.base'] = \
                -1 * (budget * df.at[idx, 'leverage']) / df.at[idx, 'sell.price'] 
            
            df.at[idx, 'b.quote'] = budget \
                + (-1 * df.at[idx, 'b.base'] * df.at[idx, 'sell.price'])

        return df

    # --------------------------------------------------------------------------
    def calculate_result_statistics(self, df:pd.DataFrame):
        
        self.start_index = 200 
        
        # df = self._calculate_balance_value(df=df)
        df = self._calculate_capital(df=df)

        df = self._calculate_max_drawdown(df=df, column='capital')
        df = self._calculate_hodl_pnl(df=df)
        df = self._calculate_max_drawdown(df=df, column='hodl.value')
        
        return df

    def _calculate_balance_value(self, df:pd.DataFrame):
        
        # df['b.value'] = self.INITIAL_CAPITAL * df['s.return']
        df.at[200, 'b.value'] = self.INITIAL_CAPITAL
        
        for idx in range(df.first_valid_index() + 1, df.last_valid_index() + 1):
            df.at[idx, 'b.value'] = \
                df.at[idx-1, 'b.value'] \
                    + df.at[idx-1, 'b.value'] * df.at[idx, 's.returns'] # * df.at[idx, 'p.size']

        return df

    def _calculate_capital(self, df:pd.DataFrame) -> pd.DataFrame:

        # df['capital'] = np.nan
        df.loc[df.first_valid_index(), 'capital'] = self.INITIAL_CAPITAL

        df.loc[
            (df['sell'] == '•'), 
            'capital'
            ] = df['b.value']
        df.loc[
            (df['buy'] == '•'),
            'capital'
            ] = df['b.value']
        
        df['capital'].ffill(inplace=True)

        return df
        
    def _calculate_max_drawdown(self, df:pd.DataFrame, column:str) -> pd.DataFrame:

        if column == 'b.value':
            df['b.max'] = df[column].expanding().max()
            df['b.drawdown'] = 1 - df['b.value'] / df['b.max']
            df['b.drawdown.max'] = df['b.drawdown'].expanding().max()
        
        if column == 'capital':
            df['cptl.max'] = df[column].expanding().max()
            df['cptl.drawdown'] = 1 - df['capital'] / df['cptl.max']
            df['cptl.drawdown.max'] = df['cptl.drawdown'].expanding().max()
            
        if column == 'hodl.value':
            df['hodl.max'] = df[column].expanding().max()
            df['hodl.drawdown'] = 1 - df['hodl.value'] / df['hodl.max']
            df['hodl.drawdown.max'] = df['hodl.drawdown'].expanding().max()
        
        return df

    def _calculate_hodl_pnl(self, df:pd.DataFrame) -> pd.DataFrame:
        
        try:
            initial_capital = self.state_params['initial_capital']            
            hodl_qty = initial_capital / df.loc[self.start_index, 'open']
            hodl_qty = round(hodl_qty, self.symbol.baseAssetPrecision)
        except KeyError as e:
            print(df.head(5))
            print(e)
            sys.exit()
            
        # print(f'calculating HODL PNL -> start index = {self.start_index}')
        # print(f'using {initial_capital} to buy {hodl_qty} {self.symbol.baseAsset}')

        df['hodl.value'] = (hodl_qty * df['close'])\
                                .round(self.symbol.quoteAssetPrecision)
                                
        df.loc[df.index < self.start_index, 'hodl.value'] = 0

        return df
  
    def _calculate_sharpe_ratio(self, 
                                df: pd.DataFrame,
                                column: str = 'b.value', 
                                trading_days: int = 365) -> float:
        
        returns = df[column] - df[column].shift()
        avg_returns = returns.mean()
        std_returns = returns.std()
        risk_free_daily = 0.0025 / 365
        
        sharpe = (avg_returns - risk_free_daily) / std_returns
        annualized_sharpe = round(sharpe * (trading_days ** 0.5), 2)
        
        return annualized_sharpe

    
    # -------------------------------------------------------------------------
    def show_overview(self, df):

        include_columns = ['human open time', 'high', 'low', 'close']
        
        if 'test' in df.columns:
            include_columns.append('test')

        for c in df.columns:
            if c.split('.')[0] == 'p': include_columns.append(c)
            
        include_columns.append('leverage')
        
        for c in df.columns:
            if c.split('.')[0] == 'buy': include_columns.append(c)
            if c.split('.')[0] == 'sell': include_columns.append(c)

        include_columns.append('sl.current')
        include_columns.append('sl.pct')
        include_columns.append('sl.trig')
        include_columns.append('sl.l.trig')

        # # include_columns.append('sl.long.trig')
        # include_columns.append('sl.long')
        # # include_columns.append('sl.short.trig')
        # include_columns.append('sl.short')
        
        include_columns.append('s.all')
                
        for c in df.columns:
            # if c.split('.')[0] == 'sl': include_columns.append(c)
            if c.split('.')[0] == 'tp': include_columns.append(c)
            if c.split('.')[0] == 'b': include_columns.append(c)

        # if 'returns.log' in df.columns:
        #     include_columns.append('returns.log')
        # if 's.returns' in df.columns:            
        #     include_columns.append('s.returns')

        
        # try:
        #     include_columns.append('returns.pct')
        # except:
        #     pass
        
        include_columns.append('position')
        include_columns.append('event.id')
        # include_columns.append('b.value')
        
        # .....................................................................
        # pd.set_option('precision', 8)
        pd.options.display.max_rows = 400
        # pd.set_option("max_rows", 400)
        # pd.set_option("min_rows", 100)

        df['sl.pct'] = df['sl.pct'] * 100
        # df['b.base'] = df['b.base'].apply(lambda x: '%.8f' % x)

        df = df.replace(np.nan, '', regex=True)
        df = df.replace(False, '', regex=True)
        df = df.replace(0, '', regex=True)
        
        
        print(df.loc[
                self.moneta.start_index:self.moneta.end_index, include_columns
                ]
            )


    def show_results(self, df):

        include_columns = ['human open time', 'close', 's.all']

        for c in df.columns:
            if c.split('.')[0] == 'buy': include_columns.append(c)
            if c.split('.')[0] == 'p': include_columns.append(c)
            if c.split('.')[0] == 'sell': include_columns.append(c)

        include_columns.append('sl.current')
        include_columns.append('sl.pct')
        include_columns.append('sl.trig')

        # include_columns.append('s.all')
                
        for c in df.columns:
            if c.split('.')[0] == 'tp': include_columns.append(c)
            # if c.split('.')[0] == 's': include_columns.append(c)
           
        include_columns.append('returns.log')
        
        try:
            include_columns.append('leverage')
        except:
            pass

            
        if 'cool.off' in df.columns: 
            include_columns.append('cool.off')

        include_columns.append('b.base')
        include_columns.append('b.quote')
        include_columns.append('b.value')
        include_columns.append('capital')
        include_columns.append('position')
        include_columns.append('cptl.drawdown')
        include_columns.append('hodl.drawdown')
        
        # .....................................................................
        # pd.set_option('precision', 8)
        # pd.set_option("max_rows", 400)
        # pd.set_option("min_rows", 100)

        for col in df.columns:
            if col in ('sl.pct', 'returns.pct', 'returns.log'):
                df[col] = df[col] * 100
                df[col] = df[col].apply(lambda x: '%.1f' % x)
                df[col].replace(np.nan, '')

        df = df.replace(np.nan, '', regex=True)
        df = df.replace(False, '', regex=True)
        df = df.replace(0, '', regex=True)
        
        print(df.loc[
                self.moneta.start_index:self.moneta.end_index, include_columns
                ].tail(50)
            )
        
        # print(df.columns)
        
    # -------------------------------------------------------------------------
    def __get_event_id(self):
        selection = ascii_lowercase + digits
        return ('').join([choice(selection) for _ in range(8)])
    
    def __delete_unnecessary_columns(self, df):
        cols = ['b.b.free', 'b.b.lock', 'b.q.free', 'b.q.lock']
        for col in cols:
            df.drop(col, axis=1, inplace=True)
            
        return df
        