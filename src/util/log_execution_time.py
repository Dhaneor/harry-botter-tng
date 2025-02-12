#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 04 00:38:20 2025

@author dhaneor
"""

import time
from functools import wraps
from util.timeops import seconds_to


def log_execution_time(logger):
    def decorator(func):
        @wraps(func)
        def execution_timer(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            human_readable_time = seconds_to(execution_time)
            logger.info("Function '%s' executed in %s", func.__name__, human_readable_time)

            return result
        return execution_timer
    return decorator