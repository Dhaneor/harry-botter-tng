#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""

from dataclasses import dataclass

# =============================================================================
@dataclass
class MarginLoanInfo:
    asset: str
    currency: str
    precision: int
    
    borrow_max_amount: float
    buy_max_amount: float
    hold_max_amount: float

    available_balance: float
    hold_balance: float
    total_balance: float
    
    liability: float
    max_borrow_size: float
    
    max_leverage: int
    liq_debt_ratio: float
    warning_debt_ratio: float
    
    def __repr__(self):
        return f'[MarginLoanInfo {self.asset}] \tliabiity={self.liability} '\
            f'(max: {self.max_borrow_size} ({self.max_leverage}x)) '\
            f'\t[BALANCE] available={self.available_balance} '\
            f'hold={self.hold_balance} total={self.total_balance} '\
            f'\t[MAX] borrow={self.borrow_max_amount}'\
            f' buy={self.buy_max_amount} hold={self.hold_max_amount}'
    
        