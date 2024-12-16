#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:57:06 2021

@author: dhaneor
"""
import sys,os
from pprint import pprint
from typing import Optional

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
from staff.hermes import Hermes
from util.timeops import execution_time
from helpers.accounting import Accounting

# ==============================================================================
class Symbol:
    """This class represents a trading symbol on the exchange and is used
    by almost every other class to access the properties of the symbol
    like restraints for orders sizes or the values for rounding prescision.
    """
    def __init__(self, symbol_name:str, data:Optional[dict]=None):

        if symbol_name is None:
            raise ValueError('Symbol name cannot be <None>!')

        self.name = symbol_name
        self.symbol_name = symbol_name
        self.exchange = self._determine_exchange_from_symbol_name()
        self._initialize(data=data)

    def __str__(self):
        permissions = (', ').join(self.permissions)

        return f'[{self.name}] on {self.exchange.upper()} '\
                f' ({permissions}) '\
                f'is currently: {self.status.lower()}'


    def _initialize(self, data:Optional[dict]=None):
        """This method gets the raw symbol information from Hermes and
        initializes all values.

        This is what we get from Hermes:
        ..code:: python
            {
                'baseAsset': 'XRP',
                'baseAssetPrecision': 8,
                'baseCommissionPrecision': 8,
                'f_icebergParts_limit': '10',
                'f_lotSize_maxQty': '90000000.00000000',
                'f_lotSize_minQty': '1.00000000',
                'f_lotSize_stepSize': '1.00000000',
                'f_marketLotSize_maxQty': '1273382.29513888',
                'f_marketLotSize_minQty': '0.00000000',
                'f_marketLotSize_stepSize': '0.00000000',
                'f_maxNumAlgoOrders': '200',
                'f_maxNumOrders': '5',
                'f_minNotional_applyToMarket': 1,
                'f_minNotional_avgPriceMins': '5',
                'f_minNotional_minNotional': '0.00010000',
                'f_percentPrice_avgPriceMins': '5',
                'f_percentPrice_multiplierDown': '0.2',
                'f_percentPrice_multiplierUp': '5',
                'f_priceFilter_maxPrice': '1000.00000000',
                'f_priceFilter_minPrice': '0.00000001',
                'f_priceFilter_tickSize': '0.00000001',
                'icebergAllowed': True,
                'isMarginTradingAllowed': True,
                'isSpotTradingAllowed': True,
                'ocoAllowed': True,
                'orderTypes': 'LIMIT,LIMIT_MAKER,MARKET,STOP_LOSS_LIMIT,\
                                TAKE_PROFIT_LIMIT',
                'permissions': 'SPOT,MARGIN',
                'quoteAsset': 'BTC',
                'quoteAssetPrecision': 8,
                'quoteCommissionPrecision': 8,
                'quoteOrderQtyMarketAllowed': True,
                'quotePrecision': 8,
                'status': 'TRADING',
                'symbol': 'XRPBTC',
                'updateTime': 1645912055
            }
        """
        if data is None:
            hermes = Hermes(exchange=self.exchange, verbose=True)
            s = hermes.get_symbol(self.name)
            if s is None:
                raise ValueError(f'{self.name} is not a valid symbol')
        else:
            s = data

        missing_values = self._check_for_missing_values(s)
        if missing_values:
            raise ValueError(f'Missing Values for {missing_values}')

        self.exchange = 'kucoin' if '-' in self.name else 'binance'

        self.baseAsset = s.get('baseAsset')
        self.baseAssetPrecision = s.get('baseAssetPrecision')
        self.baseCommissionPrecision = s.get('baseCommissionPrecision')

        self.f_icebergParts_limit = s.get('f_icebergParts_limit')

        self.f_lotSize_maxQty = s.get('f_lotSize_maxQty')
        self.f_lotSize_minQty = s.get('f_lotSize_minQty')
        self.f_lotSize_stepSize = s.get('f_lotSize_stepSize')
        self.f_stepPrecision = Accounting.get_precision(
            self.f_lotSize_stepSize
            )

        self.f_marketLotSize_maxQty = s.get('f_marketLotSize_maxQty')
        self.f_marketLotSize_minQty = s.get('f_marketLotSize_minQty')
        self.f_marketLotSize_stepSize = s.get('f_marketLotSize_stepSize')


        self.f_maxNumAlgoOrders = s.get('f_maxNumAlgoOrders')
        self.f_maxNumOrders = s.get('f_maxNumOrders')

        self.f_minNotional_applyToMarket = s.get('f_minNotional_applyToMarket')
        self.f_minNotional_avgPriceMins = s.get('f_minNotional_avgPriceMins')
        self.f_minNotional_minNotional = s.get('f_minNotional_minNotional')

        self.f_percentPrice_avgPriceMins = s.get('f_percentPrice_avgPriceMins')
        self.f_percentPrice_multiplierDown = s.get('f_percentPrice_multiplierDown')
        self.f_percentPrice_multiplierUp = s.get('f_percentPrice_multiplierUp')

        self.f_priceFilter_maxPrice = s.get('f_priceFilter_maxPrice')
        self.f_priceFilter_minPrice = s.get('f_priceFilter_minPrice')
        self.f_priceFilter_tickSize = s.get('f_priceFilter_tickSize')
        self.f_tickPrecision = Accounting.get_precision(
            self.f_priceFilter_tickSize
            )

        self.icebergAllowed = s.get('icebergAllowed')
        self.isMarginTradingAllowed = s.get('isMarginTradingAllowed')
        self.isSpotTradingAllowed = s.get('isSpotTradingAllowed')

        self.ocoAllowed = s.get('ocoAllowed')
        self.orderTypes = s.get('orderTypes')
        self.permissions = s.get('permissions')

        self.quoteAsset = s.get('quoteAsset')
        self.quoteAssetPrecision = s.get('quoteAssetPrecision')
        self.quoteCommissionPrecision = s.get('quoteCommissionPrecision')
        self.quoteOrderQtyMarketAllowed = s.get('quoteOrderQtyMarketAllowed')
        self.quotePrecision = s.get('quotePrecision')

        self.status = s.get('status')
        self.symbol = s.get('symbol')

    def _check_for_missing_values(self, symbol:dict) ->list:
        missing =  [k for k,v in symbol.items() if v is None]

        # some fields/keys were recently added by Binance, but we
        # don't need them and it doesn't matter if we don't have
        # values for these keys
        we_dont_care = ['cancelReplaceAllowed',
                        'f_trailingDelta_maxTrailingAboveDelta',
                        'f_trailingDelta_maxTrailingBelowDelta',
                        'f_trailingDelta_minTrailingAboveDelta',
                        'f_trailingDelta_minTrailingBelowDelta']

        missing = [item for item in missing if item not in we_dont_care]

        if missing:
            return missing
        return None

    def _determine_exchange_from_symbol_name(self):
        return 'kucoin' if '-' in self.name else 'binance'



# ==============================================================================
@execution_time
def test_create_symbol(symbol_name:str):
    s = Symbol(symbol_name)
    pprint(s.__dict__)

# ==============================================================================
if __name__ == '__main__':
    symbol_name = 'XLM-USDT'
    test_create_symbol(symbol_name)