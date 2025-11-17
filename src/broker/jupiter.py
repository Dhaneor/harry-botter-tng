#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 08 17:45:23 2021

@author dhaneor
"""
from typing import List, Dict, Tuple

from broker.ganesh import Ganesh
from broker.models.account import Account


""" 
TODO    add a get_position method (possibly to Ganesh) which returns
        a Position object which contains all necessary
        informations for an asset position ... this can then be used 
        by other objects to be informed about the current state/position,
        so we don't have to try to get these informations every time
        we need them. 
        
        right now, the Position/PositionFactory is part of Account. But
        then the PositionHandler cannot use it ... so move this to Ganesh
        and everyone can use it!
        
TODO    It would also be good to finish the PositionDetective to have more 
        informations about the positions
     
TODO    test all thinkable situations extensively (open long/short,
        switch between long/short, close positions, update SL and/or
        TP, reduce/increase position, ...)
"""


# ==============================================================================
class Jupiter:
    """Jupiter is our Senior Execution Manager.
    
    He takes care of all necessary changes regarding position sizes
    and stop losses. The only input he needs/gets, is a list of assets
    with their required position sizes and stop loss / take profit 
    levels. Based on this list he autonomously decides upon all necessary 
    changes and uses Ganesh to execute them.
    """
    def __init__(self, exchange:str, market:str, credentials:dict, quote_asset:str = 'USDT'):
        """
        Initializes Jupiter.

        Args:
            exchange (str): The exchange name.
            market (str): The market name.
            credentials (dict): Credentials for the broker.
            quote_asset (str, optional): The quote asset to use. Defaults to 'USDT'.
        """
        self.name: str = 'JUPITER'
        self.function: str = 'Senior Execution Manager'       
        self.exchange: str = exchange.lower()
        self.market: str = market
        self._quote_asset = quote_asset
                 
        self.broker = Ganesh(exchange=self.exchange, 
                             market=self.market,
                             credentials=credentials)

    @quote_asset.setter
    def quote_asset(self, quote_asset:str):
        valid_quote_assets = getattr(self.broker, "valid_quote_assets", ["USDT"])
        if quote_asset in valid_quote_assets:
            self._quote_asset = quote_asset
            self.account = Account(broker=self.broker, quote_asset=quote_asset) 
        else:
            raise ValueError(f'{quote_asset} is not a valid quote asset')
    
    @quote_asset.setter
    def quote_asset(self, quote_asset:str):
        if quote_asset in self.broker.valid_quote_assets:
            self._quote_asset = quote_asset
            self.account = Account(broker=self.broker, quote_asset=quote_asset) 
        else:
            raise ValueError(f'{quote_asset} is not a valid quote asset')
     
    # --------------------------------------------------------------------------
    
    def get_account(self):
        return self.account
    
    def update_account(self, target_account:List[dict]): 
        
        self.account.execute_changes(target_account)



        
        
        
        
    

        
