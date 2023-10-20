#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:54:27 2021

@author: dhaneor
"""

import pandas as pd

from pprint import pprint

from exchange.binance_classic import Binance

def get_cross_margin_asset_info(asset):

    account = binance.client.get_margin_account()
    assets = account.get('userAssets')

    free_amount = 0

    for item in assets:
        if item.get("asset") == asset:
            free_amount = float(item.get('free'))

    asset_info = binance.client.get_margin_asset(asset=asset)
    asset_name = asset_info.get('assetFullName')
    
    asset_loan_info = binance.client.get_max_margin_loan(asset=asset)
    print(asset_loan_info)

    max_loan = float(asset_loan_info.get('amount'))
    if free_amount > 0: leverage = round((max_loan + free_amount) / free_amount, 2)
    else: leverage = 'N/A'

    print('-==-'*10)
    print(f'margin asset:    {asset_name}')
    print(f'free/available:  {free_amount} {asset}')
    print(f'max loan:        {max_loan} {asset}')
    print(f'max leverage:    {leverage}')
    print('-==-'*10)


def get_isolated_account_info():

    info = binance.client.get_isolated_margin_account()

    assets = info.get('assets')
    
    print(f'Got {len(assets)} assets')
    # pprint(info)

    constructor = {'symbol' : [],
                   'ratio' : [],
                   'active' : [],
                   'created' : []
                   }

    for asset in assets:

        constructor['symbol'].append(asset.get('symbol'))
        constructor['ratio'].append(asset.get('marginRatio'))
        constructor['active'].append(asset.get('tradeEnabled'))
        constructor['created'].append(asset.get('isolatedCreated'))

    df = pd.DataFrame(constructor)
    print(df)


def get_all_margin_symbols():

    all_symbols = binance.client.get_all_isolated_margin_symbols()

    constructor = {'symbol' : [],
                   'base' : [],
                   'quote' : [],
                   'isMarginTrade' : [],
                   'isBuyAllowed' : [],
                   'isSellAllowed' : []
                   }

    for symbol in all_symbols:

        for k,v in symbol.items():

            constructor[k].append(v)
            

    df = pd.DataFrame(constructor)

    print_df = df[df['quote'] == 'BTC']
    print(print_df.tail(50))
    print(print_df.info())
    print('-==-'*50)



# =============================================================================
if __name__ == '__main__':

    binance = Binance()

    result = binance.create_stop_loss_limit_order(symbol='ADAUSDT',
                                                    side='SELL',
                                                    quantity=16.4,
                                                    stop_price=1.151,
                                                    limit_price=1.15
                                                    )

    pprint(result)

    