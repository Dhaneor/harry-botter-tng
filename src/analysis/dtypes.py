#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 06 01:28:53 2024

@author: dhaneor
"""
import numpy as np

SIGNALS_DTYPE = np.dtype([
    ('open_long', np.bool_),
    ('close_long', np.bool_),
    ('open_short', np.bool_),
    ('close_short', np.bool_)
])