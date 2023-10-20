#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 00:22:20 2022

@author dhaneor
"""
import os
import logging

# =============================================================================
from broker.config import LOG_LEVEL, LOG_DIRECTORY, LOG_FILE
from staff.zeus import Zeus

log_dir = os.path.join(
    os.path.normpath(
        os.getcwd() + os.sep + os.pardir
    ), LOG_DIRECTORY
)
log_fname = os.path.join(log_dir, LOG_FILE)

LOGGER = logging.getLogger('main')
LOGGER.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_fname)
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)

LOGGER.addHandler(ch)
LOGGER.addHandler(fh)

# =============================================================================

"""
TODO    implement a central event bus

TODO    build a web app to manage users

TODO    implement Kelly Formula for position size/leverage to use, see:
        Ernest P. Chan 'Quantitative Trading' (Kindle) on page 113

TODO    implement a mechanism that checks how much KCS we need to pay
        fees in KCS and buys some if necessary ... in AccountManager
"""

# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':

    Zeus()