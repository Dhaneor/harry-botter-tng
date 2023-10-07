#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file basic settings for ZeroMQ components.

Created on Fri Oct 06 21:41:23 2023

@author_ dhaneor
"""
DEV_ENV = True


# some default values that are (can) be shared between components
ENCODING = "utf-8"
HB_INTERVAL = 1  # seconds
HB_LIVENESS = 10  # heartbeat liveness
RGSTR_TIMEOUT = 10  # seconds
RGSTR_LOG_INTERVAL = 900  # resend request after (secs)
RGSTR_MAX_ERRORS = 10  # maximum number of registration errors

COLLECTOR_KEYS = (
    'L>&NKg9E/Cxv)nw]rXl<mgy!!w:9%s($@=Fk#DDP',
    '5sIbhID73=!fbqYBeiipw)9p0?(ix#SG]SSN$KqJ'
)

collector_sub = "tcp://127.0.0.1:5582"
collector_pub = "tcp://127.0.0.1:5583"
collector_mgmt = "tcp://127.0.0.1:5570"
collector_hb = "tcp://127.0.0.1:5580"

STREAMER_BASE_PORT = 5500