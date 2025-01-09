#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 09 00:44:23 2025

@author dhaneor
"""
import numpy as np
from abc import ABC, abstractmethod

class BaseWrapper(ABC):

    def __init__(self, data: np.ndarray):
        self.data = data

    def __call__(self) -> object:
        return self.data

    @abstractmethod
    def __getitem__(self, item):
        ...
    