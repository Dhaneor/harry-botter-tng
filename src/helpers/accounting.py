#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:03:33 2022

@author: dhaneor
"""

import sys

from decimal import Decimal
from typing import Optional, Union


class Accounting:
    """General class for money operations"""

    @classmethod
    def round(self, amount: str, ndigits: Optional[int] = 0) -> Decimal:
        """Rounds the amount using the current `Decimal` rounding algorithm."""
        if ndigits is None:
            ndigits = 0

        return Decimal(amount).quantize(ndigits)
    
    @classmethod
    def get_precision(self, value:Union[int, float, str]) ->int:
        try:
            value = str(float(value))
        except:
            raise ValueError(f'Value must int|float|str but was {type(value)}')

        if not '.' in value and not 'e' in value: 
            return 0
        
        # for scientific notation
        if 'e' in value:
            _exp = int(value.split('e')[1])
            
            if '.' in value:
                _dec = value.split('.')[1]
                _dec = len(_dec.split('e')[0])
                _corr = 1 if value.split('.')[0] == '0' else 2
            else:
                _dec = 0

            if _exp < 0:
                return _dec - _exp
            else:
                if _exp > _dec:
                    return 0
                else:
                    return _dec - _exp - _corr
            
        res = len(value.split('.')[1])
        if str(value.split('.')[1]) == '0': 
            res -= 1 

        return res

# =============================================================================
def test_precision():
    values = [4.43543, 0, '78.43', '8.3', '1.23e-03', 1.23e-03, 8.89769434e-03, 
              9.25987864874599569e+08, 1.25987864874599569e+08,
              8.45e03, '8.45e03',  4.173989e02, '1e-03', 1e-08, 1e20]
    a = Accounting

    for _v in values:
        p = a.get_precision(value=_v)
        print(f'precision for {_v} ({float(_v)}): {p}')
    
                
# =============================================================================

if __name__ == '__main__':

    test_precision()