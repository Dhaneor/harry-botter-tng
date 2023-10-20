#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:17:45 2021

@author dhaneor
"""

import sys
import time
from configparser import ConfigParser
from pprint import pprint

from broker.models.orders import ExecutionReport, MarketOrder, Order, StopLimitOrder
from models.symbol import Symbol
from _backups.balance import Balance, SpotTradingBalance


# =============================================================================
class ConfigGetter(ConfigParser):

    def __init__(self):
        
        ConfigParser.__init__(self)

    # -------------------------------------------------------------------------
    def get_config(self, symbol_name:str, interval:str):

        self._read_from_file()
        result = {}

        for k,v in self['GLOBAL'].items(): result[k] = v

        for section in self.sections():
            if section.split('_')[0] == 'SYMBOL':
                for k,v in self[section].items(): 
                    result[k] = v
                
                if result['symbol'] == symbol_name \
                and result['interval'] == interval: 

                    # values that should be set to boolean: True
                    true_values = ['True', 'true', 'yes']
                    # values that should be set to boolean: False
                    false_values = ['False', 'false', 'no']
                    
                    for k, v in result.items():
                        if v in true_values: result[k] = True
                        if v in false_values: result[k] = False

                    return result
        
        return
    
    # -------------------------------------------------------------------------
    def _read_from_file(self):

        self.read('config.ini')

# =============================================================================
class State:

    '''
    This class holds the config and the state of the bot for live access 
    and also for saving it to disk so the state can be restored/remembered
    after a restart.
    '''

    def __init__(self, bot_name:str ='Bot with no name'):

        self.name : str = bot_name
        self.status = "chillin´"

        self.balance: Balance = None

        self.exchange : str = None
        self.symbol : Symbol = None
        self.symbol_name : str = None
        self.mode : str = ''
        self.market : str = None
        self.interval : str = None
        self.leverage : float = 1
        self.INITIAL_CAPITAL : float = 0.00
        
        self.long_allowed : bool = False
        self.short_allowed : bool = False

        self.current_epoch : int = 0
        self.running_since : int = int(time.time())
        self.last_price : float = 0.00

        # ---------------------------------------
        self.base_position = {'open time' : 0,
                              'open epoch' : 0,
                              'entry price' : 0.00,
                              'amount' : 0.00,
                              'value at entry' : 0.00,
                              'value' : 0.00,
                              'pnl' : 0.00
                              }

        self.position : dict = self.base_position
        self.in_position : bool = False
        self.type_position : str = None

        # ---------------------------------------
        self.sl_active : bool = False
        self.sl_active_order : StopLimitOrder = None

        self.sl_exc_report = None # last sl order execution report
        self.last_stop_loss_order_id = None # order id of currently active sl 

        self.ready_to_go : bool = False
        self._sanity_check()

    def __repr__(self):

        out = '\n'
        out += '«=°∆°=»' * 14 + '\n\n'
        out += f'{self.name.upper()}: '
        
        if self.exchange is not None:
            out += f'We are {self.status.upper()} in the {self.market} market '
            out += f'for {self.symbol_name} ({self.interval}) on '
            out += f'{self.exchange.upper()}\n\n'

            out += f'mode: {self.mode}\n'
            out += f'longing:  {self.long_allowed}\n'
            out += f'shorting: {self.short_allowed}\n'
            out += f'leverage: {self.leverage}\n\n'
            out += f'initial capital: {self.INITIAL_CAPITAL} {self.symbol.quoteAsset}\n'
        else:
            out += f'I am lost! There was no configuration for '
            out += f'{self.symbol_name} ({self.interval})!\n\n'

        if self.in_position:

            out += f'We are in a {self.type_position} position\n\n'

        out += '«=°∆°=»' * 14 + '\n'
        return out

    # -------------------------------------------------------------------------
    def initialize(self, symbol:Symbol=None, interval:str=None, params:dict=None):

        '''
        The configuration will be read from a file with the name 'config.ini'
        which must be in the same directory as the bot. 
       
        Alternatively a dictionary with the parameters can be provided and the 
        'symbol_name' and 'interval' arguments are optional and will be 
        overridden anyways. This is useful when this class is used in a 
        backtesting context.
        
        The dictionary should look like this:

        {'exchange' : <str>,
         'symbol' : <str>,
         'mode' : <str>,
         'market' : <str>,
         'long_allowed' : bool,
         'short_allowed' : bool,
         'initial_capital' : float,
         'mode' : str
         }

        For backtesting the following values are optional: exchange, mode (will
        be set to backtesting when no value is given), so it can look like this:
        
        {'symbol' : <str>,
         'long_allowed' : bool,
         'short_allowed' : bool,
         'initial_capital' : float,
         'mode' : str
         }
        '''
        
        if symbol is not None:
            self.symbol = symbol
            self.symbol_name = self.symbol.symbol_name
            self.interval = interval

        if params is None: 
            self._initialize_from_config(symbol_name=self.symbol_name,
                                         interval=interval)
        else: 
            self._initialize_from_params(params)

        if self.mode == 'backtest':
            self.balance = SpotTradingBalance(symbol=self.symbol, 
                                initial_capital=self.INITIAL_CAPITAL)
        else:
            self.balance = Balance(symbol=self.symbol, 
                                initial_capital=self.INITIAL_CAPITAL)

    def get(self):
        pass

    def update(self, event):

        if isinstance(event, (MarketOrder, StopLimitOrder)):

            self._process_execution_report(event.last_execution_report)

    # -------------------------------------------------------------------------
    # all Execution Reports will go here for processing and taking 
    # approbriate action
    def _process_execution_report(self, er : ExecutionReport):

        if er.ex_type is None: 
            del er
            return

        if len(self.execution_reports) == 0:
            self.execution_reports.append(er)

        else:
            # print('-'*150)
            # pprint(self.execution_reports)
            # print('-'*150)
            
            for index,r in enumerate(self.execution_reports):

                if r == er:
                    self.execution_reports.pop(index)
                    er.ex_type = 0
                    # print(f'\t\tfound DUPLICATE: {er}')

            if er.ex_type == 0: 
                del er
                return

        # print(f'\n[{self.name}] no duplicate for: {er}')
        self.execution_reports.append(er)
        
        # ---------------------------------------------------------------------
        # process BUY event
        if er.ex_type == 1:
            
            if not self.position_type == 'SHORT':
                self.in_position = True
                self.position_type = 'LONG'
                self._open_position(er, 'long')
            else:
                self.in_position = False
                self.position_type = None
                self.position = self.base_position

            self.balance.trading.process_buy_event(er)
           # os.system('say "buy order executed"')

        # ---------------------------------------------------------------------
        # process SELL event (normal or because stop-loss was triggered)
        elif er.ex_type == 2:

            if not self.position_type == 'LONG':
                self.in_position = True
                self.position_type = 'SHORT'

            elif self.position_type == 'LONG':
                self.in_position = False
                self.position_type = None
                self._close_position()

            self.balance.trading.process_sell_event(er)
            # os.system('say "sell order executed"')

        # ---------------------------------------------------------------------
        # process a CANCELLED stop-loss limit order
        elif er.ex_type == 3:
            self.sl_exc_report = None
            self.last_stop_loss_order_id = None  
            self.sl_active = False
            self.balance.trading.process_sl_cancelled_event(er)

        # ---------------------------------------------------------------------
        # process NEW stop-loss order confirmation
        elif er.ex_type == 4:
            # check for duplicate as these reports are send twice
            # once when submitting the order to the system and then
            # a second time when the order is actually placed by
            # the matching engine
            duplicate = False

            # TODO ... change this to check the value for 
            # er.is_the_message_on_the_book -> no = SL created
            # yes = SL order placed (stop price reached)

            if self.sl_exc_report is not None:
                if er.client_order_id == self.sl_exc_report.client_order_id:
                    duplicate = True

            self.sl_exc_report = er
            self.last_stop_loss_order_id = er.order_id
            self.sl_active = True  
            
            if not duplicate: self.balance.trading.process_sl_created(er)   

        # ---------------------------------------------------------------------
        # process STOP LOSS triggered event
        elif er.ex_type == 5:
            self.in_position = False
            self.sl_exc_report = None
            self.position_type = None
            self.sl_active = False
            self._close_position()
            self.balance.trading.process_sl_triggered(er)
            # os.system('say "stop loss triggered!"')
            time.sleep(1)

        # ---------------------------------------------------------------------
        # self._compare_execution_report_to_pending_orders(er)
        # self.messages_to_gaia.put(self.balance) 
        # print('')
        # print(self.balance)
    # -------------------------------------------------------------------------
    # everything that needs to be changed in our state when a position
    # was opened
    def _open_position(self, er : ExecutionReport, type : str):

        # TODO ... this is only accurate for LONG position! ... enhance!

        entry_price = er.cumulative_quote_asset_transacted_quantity / er.cumulative_filled_quantity
        entry_price = round(entry_price, self.symbol.quoteAssetPrecision)

        net_qty = er.cumulative_filled_quantity - er.commission_amount
        value_after_fees = net_qty * entry_price
        value_after_fees = round(value_after_fees, self.symbol.quoteAssetPrecision)

        initial_pnl = value_after_fees - er.cumulative_quote_asset_transacted_quantity 

        self.position = {'open time' : er.event_time,
                         'open epoch' : self.epoch,
                         'entry price' : entry_price,
                         'amount' : net_qty,
                         'value at entry' : er.cumulative_quote_asset_transacted_quantity,
                         'value' : value_after_fees , 
                         'pnl' : initial_pnl
                         } 


    # -------------------------------------------------------------------------
    # initialize the state from the file 'config.ini'
    def _initialize_from_config(self, symbol_name:str, interval:str):

        cg = ConfigGetter()
        config = cg.get_config(symbol_name=symbol_name, interval=interval)
        
        if config is not None: self._set_config_values(config)
        else: print('HELP! I dunno what to do because I got no config!')
    
    # initialize the state from a dictionary with the parameters
    def _initialize_from_params(self, config:dict):
        self._set_config_values(config)

    # set the actual values from the dict provided by the caller
    def _set_config_values(self, config:dict):

        if self.symbol_name is None: 
            self.symbol = config.get('symbol')
            if isinstance(self.symbol, str):
                self.symbol = Symbol(symbol_name=self.symbol_name)
        
        self.symbol_name = self.symbol.symbol_name
        self.exchange = config.get('exchange', 'Mt. GOX')
        self.mode = config.get('mode', 'backtest') # live, simulation, backtest
        self.market = config.get('market', 'spot').upper()
        self.interval = config.get('interval')
        self.long_allowed = bool(config.get('long_allowed', False))
        self.short_allowed = bool(config.get('short_allowed', False))
        self.INITIAL_CAPITAL = float(config.get('initial_capital', 0.00))

        self.status = 'gettin´ ready'

    def _sanity_check(self):

        if self.market == 'SPOT': self.leverage = 1

        # TODO  check all values if they make sense

        self.ready_to_go = True
        

