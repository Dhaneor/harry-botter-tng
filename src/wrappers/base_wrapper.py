#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 09 00:44:23 2025

@author dhaneor
"""
import numpy as np
from typing import Sequence

class BaseWrapper:

    def __init__(self, data: np.ndarray, columns: Sequence[str]):
        self.data = data
        self.columns: list[str] = columns

    def __call__(self) -> object:
        return self.data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        if isinstance(item, str):
            for idx, col in enumerate(self.columns):
                if item == col:
                    return self.data[:, idx]
        elif isinstance(item, (slice, int)):
            return self.data[item]
        elif isinstance(item, tuple):
            return self.data[item]
        else:
            raise TypeError(f"Invalid index type ({type(item)}) for {item}")
        
    def __setitem__(self, key, value):
        if isinstance(key, str):
            for idx, col in enumerate(self.columns):
                if key == col:
                    self.data[:, idx] = value
                    return
            raise KeyError(f"Column '{key}' not found")
        elif isinstance(key, (slice, int)):
            self.data[key] = value
        elif isinstance(key, tuple):
            self.data[key] = value
        else:
            raise TypeError(f"Invalid index type ({type(key)}) for {key}")

    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim

    def mean(self) -> float:
        return np.mean(self.data)

    