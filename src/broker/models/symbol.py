#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:57:06 2021

@author: dhaneor
"""
import sys, os
from pprint import pprint
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# sys.path.append('../broker.module/')
# -----------------------------------------------------------------------------
from src.helpers.accounting import Accounting

# ==============================================================================
@dataclass
class Symbol:
    """This class represents a trading symbol on the exchange and is used
    by almost every other class to access the properties of the symbol
    like restraints for orders sizes or the values for rounding precision.
    """
    name: str
    symbol: str
    symbol_name: str
    exchange: str
    market: str
    status: str
    
    is_spot_trading_allowed: bool
    is_margin_trading_allowed: bool
    permissions: list 
    
    order_types: list
    oco_allowed: bool
    iceberg_allowed: bool
    
    iceberg_parts: int

    base_asset: str
    base_precision: int
    base_asset_precision: int
    base_commission_precision: int
    
    quote_asset: str
    quote_precision: int
    quote_asset_precision: int
    quote_commission_precision: int
    quote_order_market_qty_allowed: bool
    
    lot_size_max: float
    lot_size_min: float
    lot_size_step_size: float
    lot_size_step_precision: int
    
    market_lot_size_max: float
    market_lot_size_min: float
    market_lot_size_step_size: int
    
    min_notional: float
    min_notional_apply_to_market: bool
    min_notional_avg_price_mins: int
    
    tick_size: float
    tick_precision: int
    price_filter_min_price: float
    price_filter_max_price: float
    
    multiplier_down: float
    multiplier_up: float
    
    max_num_orders: int
    max_num_algo_orders: int
    
    def __repr__(self):
        markets = (', ').join(self.permissions)
        return f'{self.symbol} ({self.status} on {markets})'

     
class SymbolFactory:
    
    def __init__(self, exchange:str, market:str):
        self.exchange = exchange
        self.market = market
        
    def build_symbol_from_api_response(self, api_response:dict) -> Symbol:
        
        r = api_response
        f = r.get('filters')
        
        return Symbol(
            name = r.get('symbol'),
            symbol = r.get('symbol'),
            symbol_name = r.get('symbol'),
            exchange = self.exchange,
            market = r.get('permissions'),
            status = r.get('status'),
            
            is_spot_trading_allowed = r.get('isSpotTradingAllowed'),
            is_margin_trading_allowed = r.get('isMarginTradingAllowed'),
            permissions = r.get('permissions'),
            
            order_types = r.get('orderTypes'),
            oco_allowed = r.get('ocoAlllowed'),
            iceberg_allowed = r.get('icebergAllowed'),
            iceberg_parts = f[4].get('limit'),
            
            base_asset = r.get('baseAsset'),
            base_asset_precision = int(r.get('baseAssetPrecision')),
            base_precision = int(r.get('baseAssetPrecision')),
            base_commission_precision = int(r.get('baseCommissionPrecision')),
            
            quote_asset = r.get('quoteAsset'),
            quote_precision = int(r.get('quotePrecision')),
            quote_asset_precision = int(r.get('quoteAssetPrecision')),
            quote_commission_precision = int(r.get('quoteCommissionPrecision')),
            quote_order_market_qty_allowed = float(r.get('quoteOrderQtyMarketAllowed')),
            
            lot_size_max = float(f[2].get('maxQty')),
            lot_size_min = float(f[2].get('minQty')),
            lot_size_step_size = float(f[2].get('stepSize')),
            lot_size_step_precision = Accounting.get_precision(
                f[2].get('stepSize')
                ),
            
            market_lot_size_max = float(f[5].get('maxQty')),
            market_lot_size_min = float(f[5].get('minQty')),
            market_lot_size_step_size = float(f[5].get('stepSize')),
            
            min_notional = float(f[3].get('minNotional')),
            min_notional_apply_to_market = bool(f[3].get('applyToMarket')),
            min_notional_avg_price_mins = float(f[3].get('avgPriceMins')),
            
            tick_size = float(f[0].get('tickSize')),
            tick_precision = Accounting.get_precision(
                f[0].get('tickSize')
                ),
            price_filter_min_price = float(f[0].get('minPrice')),
            price_filter_max_price = float(f[0].get('maxPrice')),
              
            multiplier_down = float(f[1].get('multiplierDown')),
            multiplier_up = float(f[1].get('multiplierUp')),
                      
            max_num_orders = r.get('f_maxNumOrders', 5),
            max_num_algo_orders = r.get('f_maxNumAlgoOrders', 5)
            )
        
    def build_from_database_response(self, db_response:dict) -> Symbol:
        
        r = db_response
        
        return Symbol(
            name = r.get('symbol', ''),
            symbol = r.get('symbol', ''),
            symbol_name = r.get('symbol', ''),
            exchange = self.exchange,
            market = r.get('permissions'),
            status = r.get('status'),
            
            is_spot_trading_allowed = r.get('isSpotTradingAllowed'),
            is_margin_trading_allowed = r.get('isMARGINTradingAllowed'),
            permissions = r.get('permissions'),
            
            order_types = r.get('orderTypes'),
            oco_allowed = r.get('ocoAlllowed'),
            iceberg_allowed = r.get('icebergAllowed'),
            iceberg_parts = r.get('f_icebergParts_limit'),
            
            base_asset = r.get('baseAsset'),
            base_asset_precision = r.get('baseAssetPrecision'),
            base_commission_precision = r.get('baseCommissionPrecision'),
            
            quote_asset = r.get('quoteAsset'),
            quote_precision = r.get('quotePrecision'),
            quote_asset_precision = r.get('quoteAssetPrecision'),
            quote_commission_precision = r.get('quoteCommissionPrecision'),
            quote_order_market_qty_allowed = r.get('quoteOrderQtyMarketAllowed'),
            
            lot_size_max = r.get('f_lotSize_maxQty'),
            lot_size_min = r.get('f_lotSize_minQty'),
            lot_size_step_size = r.get('f_lotSize_stepSize'), 
            lot_size_step_precision = Accounting.get_precision(
                r.get('f_lotSize_stepSize')
                ),
            
            market_lot_size_max = r.get('f_marketLotSize_maxQty'),
            market_lot_size_min = r.get('f_marketLotSize_minQty'),
            market_lot_size_step_size = r.get('f_marketLotSize_stepSize'),
            
            min_notional = r.get('f_minNotional_minNotional'),
            min_notional_apply_to_market = r.get('f_minNotional_applyToMarket'),
            min_notional_avg_price_mins = r.get('f_minNotional_avgPriceMins'),
            
            tick_size = r.get('priceFilter_tickSize'),
            tick_precision = Accounting.get_precision(
                r.get('f_priceFilter_tickSize')
                ),
            price_filter_min_price = r.get('priceFilter_minPrice'),
            price_filter_max_price = r.get('price_Filter_maxPrice'),
              
            multiplier_down = r.get( 'f_percentPrice_multiplierDown'),
            multiplier_up = r.get( 'f_percentPrice_multiplierUp'),
                      
            max_num_orders = r.get('f_maxNumOrders'),
            max_num_algo_orders = r.get('f_maxNumAlgoOrders')
            )
        
