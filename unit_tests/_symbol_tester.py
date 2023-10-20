#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""

import sys
import os
from pprint import pprint

# -----------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from staff.hermes import Hermes
from broker.models.symbol import SymbolFactory

# ==============================================================================


# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    
    exchange = 'kucoin'
    market = 'CROSS MARGIN'
    symbol = 'BTCUSDT'
    
    hermes = Hermes(exchange=exchange, verbose=False)    
    sf = SymbolFactory(exchange=exchange, market=market)
    s = sf.build_symbol_from_api_response(hermes.get_symbol(symbol))
    
    pprint(s.__dict__)
