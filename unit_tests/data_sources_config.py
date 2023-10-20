#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:44:23 2022

@author_ dhaneor
"""
import logging
import os
import sys

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
handler.setFormatter(formatter)

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

from src.data_sources import zmq_config as cnf  # noqa: E402, F401


def test_streamer_config():

    for service_type in ("streamer", "collector"):

        exchange = 'kucoin'
        market = 'spot'
        sock_defs = []
        external_ip = '192.168.1.1'

        config = cnf.get_config(
            service_type, exchange, market,
            sock_defs=sock_defs, external_ip=external_ip
        )

        assert config.exchange == exchange
        assert config.market == market
        assert config._sock_defs == sock_defs
        assert config.service_type == service_type, \
            f"{config.service_type} != {service_type}"
        assert config.external_ip == external_ip, \
            f"{config.external_ip}!= {external_ip}"

        print(f"{service_type} --> {config.as_json()}")
        print('-------------------------------------------------------\n')

    return config


if __name__ == '__main__':
    test_streamer_config()

    print('\nAll tests passed!\n')
