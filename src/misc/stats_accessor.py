#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 00:34:23 2025

@author dhaneor
"""
import numpy as np

class StatsAccessor:
    def __init__(self, wrapper):
        self._wrapper = wrapper
    
    def mean(self) -> float:
        return np.mean(self._wrapper.data)
    
    def std(self) -> float:
        return np.std(self._wrapper.data)