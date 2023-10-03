#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

import sys
import os
import time
from typing import Union
from tqdm import tqdm

from pprint import pprint

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from models.state import State
from _backups.balance import SpotTradingBalance
from models.symbol import Symbol
from analysis.indicators import Indicators
from staff.moses import Moses
from broker.models.orders import ExecutionReport, MarketOrder, StopLimitOrder
from mock_responses.order_results import OrderResultConstructor
from staff.saturn import Saturn
from staff.moneta import Moneta
from staff.shakti import Shakti
from plotting.minerva import BacktestChart


# =============================================================================
class Backtest:
    '''This class runs a single backtest (1 symbol, 1 interval)'''

    def __init__(self, symbol:Symbol, interval:str, start:Union[int,str], 
                 end:Union[int,str], leverage:float, strategy:str,
                 state_params:dict, 
                 stop_loss_strategy:str, stoploss_params:dict,
                 verbose:bool=False, draw_chart:bool=True):

        self._ex_time_start: float = time.time()
        self._ex_time_end: float = 0
        self.execution_time: float = 0
 
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        
        self.state_params = state_params
        self.stoploss_params = stoploss_params

        self.fees = 0.001

        # initialize our helper modules
        self.state = State('The Ancient One')
        self.moses = Moses()
        self.moses.set_sl_strategy(stop_loss_strategy, stoploss_params)
        self.moneta = Moneta(symbol=self.symbol, 
                             interval=self.interval,
                             strategy=strategy,
                             start=start, end=end,
                             initial_capital=self.state_params['initial_capital'])

        self.saturn = Saturn(symbol=self.symbol)
        self.exchange = OrderResultConstructor(symbol=self.symbol)
        self.shakti = Shakti()

        # initialize yourself
        self.initialize()

        # -----------------------------
        self.current_epoch: int = 0, 
        self.last_close: float = 0.00
        self.cool_off_timer: int = 0

        self.epoch_data, self.last_epoch_data = {}, {}

        self.draw_chart = draw_chart
        self.verbose: bool = verbose
        
        self.cool_off: int = 0
        self.leverage = leverage
        self._value_for_position_sizing = self.state_params['initial_capital']
        self.risk_amount_fix = self.state_params['initial_capital'] * 2
        self.risk_percentage = 5

    @property
    def current_sl_pct(self):
        # index = self.moneta.current_index - self.start_index -1
        return self.moneta.data.loc[self.moneta.current_index-1, 'sl.l.pct']

    # -------------------------------------------------------------------------
    def initialize(self):

        # initialize the State class instance
        self.state.initialize(symbol=self.symbol, interval=self.interval, \
                              params=self.state_params)
        
        self.moneta.data = self.moses.get_stop_loss_prices(self.moneta.data)

    def run(self):

        self.start_index = self.moneta.start_index
        self.end_index = self.moneta.end_index + 1
        # print(f'_'*135)
        # print(f'starting at index [{self.start_index}]\n\n')
        self.state.status = 'workin hard´'
        
        # self.moneta.data['p.size.alt'] = self.shakti.get_position_size_by_volatility(
        #     df=self.moneta.data, risk_amount=500)

        self._get_epoch_data()
        self._copy_balance_to_epoch()
        self._save_current_epoch_to_dataframe()

        if self.verbose:
            for idx in range(self.start_index, self.end_index): 
                self.main(idx=idx)
        else:
            for idx in tqdm(range(self.start_index, self.end_index), desc="Processing..."):
                self.main(idx=idx)

        self.state.status = 'chillin´ hard'
        self.moneta.show_overview()
        print('-'*150)
        print(self.state.balance)
        # print(self.state)
        print('-'*150)

        self.moneta.calculate_result_statistics()
        self.moneta.print_statistics()

        # ---------------------------------------------------------------------
        # get and print the execution time 
        self._ex_time_end = time.time()
        self.execution_time = round(self._ex_time_end - self._ex_time_start, 2)
        out = '-'*200 + '\n'
        out += f'execution time: {self.execution_time}' 
        print(out)
        
        # ---------------------------------------------------------------------
        # create and show the chart
        if self.draw_chart:
            df = self.moneta.data \
                    .iloc[self.moneta.start_index:self.moneta.end_index, :]\
                    .copy(deep=True)
            
            chart_name = f'{self.symbol.symbol_name} ({self.interval})'
            chart = BacktestChart(df=df, title=chart_name, color_scheme='night')
            chart.draw()
    
    def main(self, idx:int):

        self.moneta.next_epoch()
        self.current_epoch = self.moneta.epoch
        self._get_epoch_data()
        self._get_last_epoch_data() 

        # -----------------------------------------------------------------
        if idx > self.start_index: 
            self._check_stop_loss_triggered()
            self._save_last_epoch_to_dataframe()

        out = '•'*200 + '\n'
        out += f'EPOCH {self.current_epoch} '
        out += f" - open time {self.epoch_data['human open time']}\n"
        self.talk(out)

        if idx == self.end_index-1:
            self._process_sell_signal()
            self._save_current_epoch_to_dataframe()
            return

        # -----------------------------------------------------------------                   
        signal = self.last_epoch_data['s.all']

        if self.cool_off_timer == 0:
            
            self._calculate_max_position_size()      
            
            if signal > 0: self._process_buy_signal()
            elif signal == 0: self._process_no_action()
            elif signal < 0: self._process_sell_signal()
        else:
            self.cool_off_timer -= 1
            self._copy_balance_to_epoch()
            
        self._save_current_epoch_to_dataframe()


    # -------------------------------------------------------------------------
    def _process_no_action(self):

        if self.state.in_position:
            self._calculate_max_position_size() 
            self._update_stop_loss()

        self._copy_data_from_last_epoch()

    def _process_buy_signal(self):
        
        self._calculate_max_position_size() 

        if self.state.in_position and self.state.type_position == 'LONG':
            self._update_stop_loss()
            self._copy_data_from_last_epoch()

        if self.state.in_position and self.state.type_position == 'SHORT':

            # TODO  implement handling of SHORT positions
            pass

        elif not self.state.in_position:
            self._open_long_position()

    def _process_sell_signal(self):

        if self.state.in_position and self.state.type_position == 'SHORT':
            # TODO  implement handling of SHORT positions
            pass

        elif self.state.in_position and self.state.type_position == 'LONG':
            self._close_position()

        else:
            self._update_balance()
            

    # -------------------------------------------------------------------------
    # methods for handling the opening and closing of positions (long/short)
    def _open_long_position(self):
        self._value_for_position_sizing = self.state.balance.quote_asset_balance['free']
        self.buy(base_qty=self.epoch_data['p.max'] * 0.998)
        
    def _update_long_position(self):
        pass
        
    def _close_position(self):

        # 'cancel' active stop-loss order
        base_qty = self.state.sl_active_order.base_qty

        self.state.sl_current = None
        self.state.sl_active_order = None
        self.epoch_data['sl.current'] = float('NaN')
        self.epoch_data['sl.pct'] = float('NaN')

        self.state.balance.process_sl_cancelled_event()

        if self.state.type_position == 'LONG':
            # sell what we have
            order = MarketOrder(symbol=self.symbol,
                                market='SPOT',
                                side='SELL',
                                base_qty=base_qty,
                                last_price=self.last_close)

        elif self.state.type_position == 'SHORT':

            # buy back the amount we shorted
            order = MarketOrder(symbol=self.symbol,
                                market='SPOT',
                                side='BUY',
                                base_qty=base_qty,
                                last_price=self.last_close)

        order = self.saturn.cm.check_order(order)

        if order.status == 'APPROVED':
            order.execution_report = self.exchange.execute(order)

        if order.status == 'FILLED':
            self._update_position_values(close_position=True)
            self._update_balance(execution_report=order.execution_report)

            if order.side == 'SELL':
                self.epoch_data['sell'] = True
                self.epoch_data['sell.price'] = order.fill_price
                self.epoch_data['sell.amount'] = order.base_qty 

                self.epoch_data['p.actv'] = False
                self.epoch_data['p.type'] = None

            elif order.side == 'BUY':
                self.epoch_data['buy'] = True
                self.epoch_data['buy.price'] = order.fill_price
                self.epoch_data['buy.amount'] = order.base_qty - order.commission_base 

                self.epoch_data['p.actv'] = True
                self.epoch_data['p.type'] = 'LONG'
                
        self.cool_off_timer = self.cool_off

    def _update_stop_loss(self):
        
        sl_type = self.stoploss_params.get('type')
        
        if self.stoploss_params.get('type', 'fixed') == 'fixed':
            self.epoch_data['sl.current'] = self.state.sl_active_order.stop_price
            return
            
        if self.stoploss_params.get('type', 'fixed') == 'breakeven':
            
            # trigger = self.state.position['entry price'] \
            #             * (1 + self.stoploss_params['percent'])
            trigger = self._get_stop_loss_price()
            
            if self.last_epoch_data['close'] >= trigger:
                sl_price = self.state.position['entry price'] * (1+self.fees)
                sl_percent = 1 - (sl_price / self.epoch_data['close'])
                self.epoch_data['sl.current'] = sl_price
            else:
                self.epoch_data['sl.current'] = self.last_epoch_data['sl.current']
                return
            
        if self.stoploss_params.get('type', 'fixed') == 'trailing':
            sl_price = self._get_stop_loss_price()
            sl_percent = round(1 - self.last_epoch_data['close'] / sl_price, 2) 
            # sl_price = self.last_epoch_data['close'] * (1-sl_percent)
            self.epoch_data['sl.pct'] = sl_percent
                
        sl_old = self.state.sl_active_order

        # ---------------------------------------------------------------------
        if sl_price > sl_old.stop_price:
            sl_order = StopLimitOrder(symbol=self.symbol,
                                    market=self.state.market,
                                    side='SELL',
                                    base_qty=sl_old.base_qty,
                                    stop_price=sl_price,
                                    limit_price=sl_price*0.95,
                                    last_price=self.last_epoch_data['close'])

            if self.verbose: self.talk(sl_order)
            self.saturn.cm.check_order(sl_order)
            if self.verbose: self.talk(sl_order)
            result = self.exchange.execute(order=sl_order)
            sl_order.execution_report = {'success' : True, 'message' : result}
            if self.verbose: self.talk(sl_order)

            self.state.sl_active_order = sl_order
            self.epoch_data['sl.current'] = float(sl_order.stop_price)
            self.epoch_data['sl.pct'] = sl_percent

            self.state.balance.process_sl_cancelled_event()
            self._update_balance(sl_order.execution_report)

        else:
            self._copy_balance_to_epoch()
            self.epoch_data['sl.current'] = self.last_epoch_data['sl.current']

        if self.verbose: self.talk(self.state.balance)


    def _calculate_max_position_size(self, type:str='LONG'):

        _restricted = False

        # calculate the current portfolio value
        quote_amount = self.state.balance.quote_asset_balance['free'] \
            + self.state.balance.quote_asset_balance['locked']
        base_amount = self.state.balance.base_asset_balance['free'] \
            + self.state.balance.base_asset_balance['locked']
            
        last_price = self.epoch_data['close']
        sl_percent = self._get_stop_loss_price() / last_price
        
        portfolio_value = quote_amount + base_amount * last_price
        # portfolio_value = self._value_for_position_sizing 
        risk_amount = portfolio_value * self.risk_percentage / 100
        risk_amount = min(risk_amount, self.risk_amount_fix)
                
        # ----------------------------------------------------------------------            
        risk = self.last_epoch_data['close'] * self.last_epoch_data['ret.pct.avg']
        # risk = self.last_epoch_data['atr']
        
        if type == 'LONG':
            
            pos_size = self.shakti.get_max_position_size(
                atr=risk, 
                portfolio_value=portfolio_value,
                risk_amount = risk_amount,
            )
            
            pos_size_as_quote = pos_size * last_price
            borrow_amount = pos_size_as_quote - portfolio_value
            borrow_amount = max(0, borrow_amount)
            leverage = self._calculate_leverage(portfolio_value, borrow_amount)
            
            if leverage > self.leverage:
                pos_size = (portfolio_value * self.leverage * 0.998)\
                    / last_price
                leverage = self.leverage
                _restricted = True
            
            if self.verbose:
                
                print(f'{portfolio_value=} : {risk_amount=} : {risk=}')
                print(f'{pos_size=} : {pos_size_as_quote=} : {leverage=}')
                print(f'{_restricted=}')
                
            self.epoch_data['p.max'] = pos_size
            self.epoch_data['leverage'] = leverage
  
    def _calculate_leverage(self, portfolio_value:float, 
                            borrow_amount:float) -> float:
        
        return round((portfolio_value + borrow_amount) / portfolio_value, 2)
  
    def buy(self, base_qty:float):

        order = MarketOrder(symbol=self.symbol,
                            market='SPOT',
                            side='BUY',
                            base_qty=base_qty,
                            last_price=self.last_close)   

        order = self.saturn.cm.check_order(order)

        # ---------------------------------------------------------------------
        if order.status == 'APPROVED':
            order.execution_report = self.exchange.execute(order)

        if order.status == 'FILLED':

            self.state.in_position = True
            self.state.type_position = 'LONG'

            self.epoch_data['buy'] = True
            self.epoch_data['buy.price'] = order.fill_price
            self.epoch_data['buy.amount'] = order.base_qty - order.commission_base 

            self.epoch_data['p.actv'] = True
            self.epoch_data['p.type'] = 'LONG'
            
            self._update_position_values(order.execution_report)
            self._update_balance(order.execution_report)
            

            # -----------------------------------------------------------------
            # create a STOP LOSS LIMIT order for the position we just opened
            if self.stoploss_params.get('type', 'fixed') == 'fixed':
                sl_pct = float(self.stoploss_params.get('percent')) / 100
            elif self.stoploss_params.get('type', 'fixed') == 'breakeven':
                sl_pct = float(self.stoploss_params.get('percent')) / 100
            else:
                sl_pct = float(self.stoploss_params['percent']) / 100
            
            # stop_price = order.fill_price * (1-sl_pct)
            stop_price = self._get_stop_loss_price()

            limit_price = stop_price * 0.95
            base_qty = self.state.balance.base_asset_balance['free']

            sl_order = StopLimitOrder(symbol=self.symbol,
                                      market=self.state.market,
                                      side='SELL',
                                      base_qty=base_qty,
                                      stop_price=stop_price,
                                      limit_price=limit_price,
                                      last_price=order.fill_price)

            self.saturn.cm.check_order(sl_order)
            result = self.exchange._get_stop_limit_order_result(sl_order)
            sl_order.execution_report = {'success' :  True, 'message' : result}
            
            # update the state to reflect the postion and the stop-loss
            self.epoch_data['sl.current'] = float(sl_order.stop_price)
            self.epoch_data['sl.pct'] = sl_pct
            self.state.sl_active = True
            self.state.sl_active_order = sl_order
            
            self._update_balance(sl_order.execution_report)
        
        # ---------------------------------------------------------------------
        elif order.status == 'REJECTED':
            pass

    # ------------------------------------------------------------------------#
    #                               HELPER METHODS                            #
    # ------------------------------------------------------------------------#
    def _get_stop_loss_price(self):
        return self.last_epoch_data['sl.long']
    
    def _check_stop_loss_triggered(self):
        
        # ---------------------------------------------------------------------
        # LONG position
        if self.state.in_position and self.state.type_position == 'LONG':
            if self.state.sl_active_order.stop_price >= self.last_epoch_data['low']:

                if self.verbose:
                    out = f'STOP LOSS triggered in epoch {self.current_epoch-1} ({self.moneta.current_index})\n'
                    out += f"close {self.last_epoch_data['low']} \n"
                    out += f"is lower than our stop price {self.state.sl_active_order.stop_price}"
                    self.talk(out)

                sl_order = self.state.sl_active_order
                sl_order.last_price = sl_order.limit_price
                res = self.exchange._get_stop_limit_order_result(order=sl_order, 
                                                                 filled=True)
                sl_order.execution_report = {'success' : True, 'message' : res}

                # -------------------------------------------------------------
                # update our state
                self.last_epoch_data['sl.trig'] = True
                self.last_epoch_data['sell'] = True
                self.last_epoch_data['sell.price'] = sl_order.fill_price
                self.last_epoch_data['sell.amount'] = sl_order.execution_report.cumulative_filled_quantity

                self.last_epoch_data['p.actv'] = False
                self.last_epoch_data['p.type'] = None

                self.state.in_position = False
                self.state.position_type = None

                self._update_balance(sl_order.execution_report)
                self._save_last_epoch_to_dataframe()

                if self.verbose:
                    self.talk(sl_order)
                    self.talk(self.state.balance)
                    self.talk('-'*150)
                    
                self.cool_off_timer = self.cool_off

        
        # ---------------------------------------------------------------------
        elif self.state.in_position and self.state.type_position == 'SHORT':
            if self.state.sl_active_order.stop_price <= self.self.last_epoch_data['high']:
                self.last_epoch_data['sl.trig'] = True
                self.state.in_position = False
                self.state.position_type = None               
                self.cool_off_timer = self.cool_off

    def _update_balance(self, execution_report:ExecutionReport=None):

        # self.talk(f'[UPDATE BALANCE] using execution report:')
        # self.talk(execution_report)
        # self.talk(f'quantity: {execution_report.order_quantity}')

        if execution_report is not None:

            if execution_report.ex_type == 1:
                self.state.balance.process_buy_event(execution_report)
            elif execution_report.ex_type == 2:
                self.state.balance.process_sell_event(execution_report)
            elif execution_report.ex_type == 4:
                self.state.balance.process_sl_created(execution_report)
            elif execution_report.ex_type == 5:
                self.state.balance.process_sl_triggered(execution_report)
        
            if execution_report.ex_type == 5:
                self.last_epoch_data['b.b.free'] = self.state.balance.base_asset_balance['free']
                self.last_epoch_data['b.b.lock'] = self.state.balance.base_asset_balance['locked']
                self.last_epoch_data['b.q.free'] = self.state.balance.quote_asset_balance['free']
                self.last_epoch_data['b.q.lock'] = self.state.balance.quote_asset_balance['locked']

        self._copy_balance_to_epoch()
        return

    def _update_position_values(self, execution_report:ExecutionReport=None,
                                      close_position:bool=False):

        if close_position: 
            self.state.in_position = False
            self.state.type_position = None
            self.state.position = self.state.base_position
            return

        if execution_report is not None:
            
            net_qty = execution_report.cumulative_filled_quantity
            if execution_report.side == 'BUY':
                net_qty -= execution_report.commission_amount
            net_qty = round(net_qty, self.symbol.baseAssetPrecision)

            value_at_entry = round(net_qty * execution_report.last_executed_price, \
                                    self.symbol.quoteAssetPrecision)

            position_type = 'LONG' if execution_report.side == 'BUY' else 'SHORT' 

            self.state.in_position = True
            self.state.position = {'open time' : execution_report.transaction_time,
                                    'open epoch' : self.current_epoch,
                                    'type' : position_type,
                                    'entry price' : execution_report.order_price,
                                    'amount' : net_qty,
                                    'value at entry' : value_at_entry,
                                    'value' : value_at_entry,
                                    'pnl' : 0.00}

        else:
            self.state.position['value'] = round( self.state.position['net_qty'] \
                                                + self.epoch_data['close'],
                                                self.symbol.quoteAssetPrecision)

            self.state.position['pnl'] = round(self.state.position['value'] \
                                                - self.state.position['value_at_entry'], \
                                                self.quoteAssetPrecision)

    # -------------------------------------------------------------------------
    def _copy_balance_to_epoch(self):
        
        self.epoch_data['b.b.free'] = self.state.balance.base_asset_balance['free']
        self.epoch_data['b.b.lock'] = self.state.balance.base_asset_balance['locked']
        self.epoch_data['b.q.free'] = self.state.balance.quote_asset_balance['free']
        self.epoch_data['b.q.lock'] = self.state.balance.quote_asset_balance['locked']

    def _get_epoch_data(self):

        self.epoch_data = self.moneta.epoch_data

    def _get_last_epoch_data(self):

        self.last_epoch_data = self.moneta.last_epoch_data
        self.last_close = self.last_epoch_data['close']

    def _copy_data_from_last_epoch(self):

        update_columns = ['p.actv', 'p.type', 
                            'b.b.free', 'b.b.lock', 'b.q.free', 'b.q.lock']

        for col in update_columns:
            self.epoch_data[col] = self.last_epoch_data[col]

    # -------------------------------------------------------------------------
    def _save_current_epoch_to_dataframe(self):

        self.moneta.epoch_data_to_dataframe(data=self.epoch_data, 
                                            index=self.moneta.current_index)

    def _save_last_epoch_to_dataframe(self):

        self.moneta.epoch_data_to_dataframe(data=self.last_epoch_data, 
                                            index=self.moneta.current_index-1)

    # -------------------------------------------------------------------------
    def talk(self, message):

        if not self.verbose: return
        else: print(message)