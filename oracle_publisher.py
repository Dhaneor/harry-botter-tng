#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 22 12:305:20 2023

@author dhaneor
"""
import sys, os
import asyncio
import zmq, zmq.asyncio
import logging
import pandas as pd
import dataclasses
import json

from src.analysis.oracle import OracleSignal

LOG_LEVEL = "DEBUG"
logger = logging.getLogger('main')
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# -----------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# -----------------------------------------------------------------------------
ctx = zmq.asyncio.Context()

publisher_address = 'tcp://*:5555'
publisher = ctx.socket(zmq.XPUB)
publisher.bind(publisher_address)

# ==============================================================================
async def oracle_publisher():
    logger.info("Starting Oracle Publisher")

    while True:
            signals = [
                OracleSignal(
                    symbol = 'BTC-USDT',
                    weight = 0.7,
                    strategy = 'Pure Breakout',
                    data = None,
                    signal = None,
                    target_position = None,
                    target_leverage = None,
                    stop_loss = None,
                    take_profit = None  
                ),
                OracleSignal(
                    symbol = 'ETH-USDT',
                    weight = 0.3,
                    strategy = 'Pure Breakout',
                    data = None,
                    signal = None,
                    target_position = None,
                    target_leverage = None,
                    stop_loss = None,
                    take_profit = None  
                )
            ]
            
            json_data = [
                json.dumps(dataclasses.asdict(signal)).encode('utf-8') \
                    for signal in signals
            ]

            # Publish each JSON data as a separate part of the multipart message
            for data in json_data:
                await publisher.send_multipart([b'', data])
                logger.info(f'sending {data}')

            # Send the termination signal
            await publisher.send_multipart([b'', b'END'])
            
            # don't push it ...  
            await asyncio.sleep(2)

async def main():
    try:
        await oracle_publisher()
    except KeyboardInterrupt:
        ctx.term()
        
              
if __name__ == '__main__':
    
    asyncio.run(main())
