#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:48:33 2024

@author dhaneor
"""

import logging
import sys


def get_logger(name: str = "main", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with a predefined format.

    :param name: Name of the logger.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d - "
            "[%(levelname)s]: %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
