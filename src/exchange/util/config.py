#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 17:45:23 2022

@author_ dhaneor
"""
import logging

DB_HOST = "127.0.0.1"
DB_USER = "developer"
DB_PASSWORD = "123456"

# master account
API_KEY = "61fee216e52c92000137332a"
API_SECRET = "4c47b3fa-9c71-4a54-bfcf-ef4dad3c12e0"
API_PASSPHRASE = "muzick@093"

# sub-account: MerlinUSDT01
# API_KEY = "63713b16a09d220001152611"
# API_SECRET = "9e871628-3c30-478c-ae2b-e633b667cb83"
# API_PASSPHRASE = "merlin@2022"

CREDENTIALS = {'api_key' : API_KEY,
               'api_secret' : API_SECRET,
               'api_passphrase' : API_PASSPHRASE
               }

ACCOUNTS = {
    'one' : {'name': 'one',
             'is active': True,
             'user id': 1000,
             'credentials': CREDENTIALS,
             'exchange': 'kucoin',
             'market': 'cross margin',
             'symbols': ('ETH-USDT',),
             'assets': ('XRP',),
             'quote_asset': 'USDT',
             'interval': '1m',
             'strategy': 'none',
             'risk level': 2,
             'max leverage': 4,
    }   
}
 # ============================================================================
 # Log level (= Python log levels)
LOG_LEVEL = logging.DEBUG

# Name of the log directory and file. The directory must exist one level 
# above the working directory for the bot. This makes sure that we can
# replace the code anytime we want without losing the log files.  
LOG_DIRECTORY = 'logs'
LOG_FILE = 'bot_log.log'

# Minimum position change to actually trade. This is the threshhold
# that needs to be exceeded by a postion change request, before the
# bot will actually act upon the request. This is to prevent a lot
# of very small adjustments when CONTINOUS_TRADING is also set to 
# 'True' in the trading strategy that is used.
MIN_POS_CHANGE_PERCENT = 10

# These two values determine what is considered to be a valid exchange/
# market. As of now, only the CROSS MARGIN market on KUCOIN is 
# implemented. But as I plan to extend the bot's capabilities to more 
# exchanges and possibly SPOT/FUTURES trading, I've alrfeady built in 
# sanity checks that use these values here. When more implementations
# are done, we can just extend these two lists as approbriate. 
VALID_EXCHANGES = ['kucoin']
VALID_MARKETS = ['cross margin']

# The risk level can be set by the user and for now we have risk levels
# from 1 to 5. The value that is set here will determine the max risk 
# level that can be chosen when configuring an exchange account.
MAX_RISK_LEVEL = 5

# Additionally to the risk level, the user can determine that a certain
# leverage is not exceeded. This supersedes the dynamically set leverage 
# that is allowed by the risk management system, but naturally it can't
# be higher than the maximum leverage allowed by the exchange. 
# Low values for <risk level>, as set by each individual user, will 
# almost always choose very low leverage anyway (depending on the 
# volatility of the asset).  
MAX_USER_LEVERAGE = 5

# Set minimum and maximum values for the intial capital allowed when 
# configuring a new exchange account. These are only soft boundaries
# and, from the point of view of the bot, nothing bad happens if they 
# are not respected. But it's the user that may suffer or be 
# disappointed, so we prevent configuring accounts with less or more 
# capital than set here.
MIN_INITIAL_CAPITAL = 0.000_001
MAX_INITIAL_CAPITAL = 1_000_000
