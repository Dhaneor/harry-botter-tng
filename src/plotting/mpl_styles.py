#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:28:53 2021

@author: dhaneor
"""


schemes = {
    "day": {
        "canvas": "antiquewhite",
        "background": "oldlace",
        "tick": ("darkgoldenrod", 1),
        "buy": ("lime", 0.7),
        "sell": ["crimson", 0.7],
        "bull": ("limegreen", 0.5),
        "bear": ("indianred", 0.5),
        "flat": ("darkgray", 0.5),
        "position": ("burlywood", 0.2),
        "line1": ("darkgoldenrod", 0.3),
        "line2": ("darkorange", 0.5),
        "line3": ("crimson", 0.5),
        "line4": ("darkred", 0.5),
        "channel": ("peru", 0.5),
        "channel_bg": ("goldenrod", 0.15),
        "hline": ("lightgrey", 1),
        "grid": ("darkgoldenrod", 0.5),
    },
    "day2": {
        "canvas": "antiquewhite",
        "background": "linen",  # "oldlace",
        "tick": ("darkgoldenrod", 1),
        "buy": ("lime", 1),
        "sell": ["crimson", 0.7],
        "bull": ("limegreen", 0.7),
        "bear": ("indianred", 0.7),
        "flat": ("darkgray", 0.7),
        "position": ("burlywood", 0.2),
        "line1": ("blue", 0.3),
        "line2": ("crimson", 0.5),
        "line3": ("darkred", 0.5),
        "line4": ("darkorange", 0.5),
        "channel": ("peru", 0.5),
        "channel_bg": ("goldenrod", 0.15),
        "hline": ("lightgrey", 1),
        "grid": ("darkgoldenrod", 1),
    },
    "night": {
        "canvas": "#121212",
        "background": "#090004",
        "tick": ("#5f616A", 0.5),
        "buy": ("limegreen", 0.9),
        "sell": ("crimson", 0.9),
        "bull": ("chartreuse", 0.5),
        "bear": ("palevioletred", 0.5),
        "flat": ("grey", 0.7),
        "position": ("blanchedalmond", 0.1),
        "line1": ("#F7B538", 0.5),
        "line2": ("#DB7C26", 0.5),
        "line3": ("#D8572A", 0.5),
        "line4": ("#C32F27", 0.5),
        "channel": ("#F7B538", 0.7),
        "channel_bg": ("#F7B538", 0.25),
        "hline": ("lightgrey", 1),
        "grid": ("darkgrey", 0.4),
    },
}
