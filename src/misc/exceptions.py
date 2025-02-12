#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 16 01:57:23 2025

@author dhaneor
"""

class DimensionMismatchError(Exception):
    """Raised when there's a mismatch in dimensions during data assignment."""

    pass