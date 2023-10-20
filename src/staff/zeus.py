#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:47:20 2022

@author dhaneor
"""
import time
import logging

from typing import List, Union
from multiprocessing import Queue

from staff.account_manager import AccountManager
from staff.thoth import Thoth
from models.users import Account
from analysis.util.ohlcv_observer import OhlcvObserver
from analysis.oracle import LiveOracle

logger = logging.getLogger('main.zeus')
oracle = LiveOracle()
notify_queue = Queue()
thoth = Thoth(event_queue=notify_queue)

# =============================================================================
class Zeus:

    def __init__(self):

        self.ohlcv_observer = OhlcvObserver()

        self.accounts: List[Account] = self.get_accounts()
        self.account_managers: List[AccountManager] = []
        self.start_account_managers()

    # -------------------------------------------------------------------------
    def run(self):
        pass

    # -------------------------------------------------------------------------
    def get_accounts(self) -> List[Union[Account, None]]:

        def _build_from_dict(name, item):
            return Account(name=name,
                           is_active=item['is active'],
                           api_key=item['credentials']['api_key'],
                           api_secret=item['credentials']['api_secret'],
                           api_passphrase=item['credentials']['api_key'],
                           date_created=int(time.time()),
                           exchange=item['exchange'],
                           market=item['market'],
                           interval=item['interval'],
                           assets=item['assets'],
                           quote_asset=item['quote_asset'],
                           strategy=item['strategy'],
                           risk_level=item['risk level'],
                           max_leverage=item['max leverage'],
                           id=item['user id'],
            )

        from broker.config import ACCOUNTS

        return [_build_from_dict(name, item) for name, item in ACCOUNTS.items()]

    def start_account_managers(self):
        for acc in self.accounts:
            try:
                AccountManager(
                    account=acc,
                    ohlcv_observer=self.ohlcv_observer,
                    oracle=oracle,
                    notify_queue=notify_queue
                )
            except Exception as e:
                logger.exception(e)
