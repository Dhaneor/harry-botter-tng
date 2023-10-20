#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:32:20 2022

@author dhaneor
"""
import sys, time
import zmq
import zmq.asyncio
import asyncio
import datetime as dt
import logging
import json

logger = logging.getLogger('oracle_listener')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
)
ch.setFormatter(formatter)

logger.addHandler(ch)

from typing import Callable, Union


publisher_address = 'tcp://127.0.0.1:5555'

# ==============================================================================
async def subscriber(ctx: zmq.asyncio.Context,
                       address: str, topic: str = 'signal',
                       callback: Union[Callable, None]=None):
            
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(address)
    
    if topic != '':
        subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode())
    else:
        subscriber.setsockopt(zmq.SUBSCRIBE, b'')
            

    logger.info(f'subscriber started for topic {topic}')

    count = 0
    start = time.time()
    while True:
        
        msg = [None, None]
        
        try:
            
            msg = await subscriber.recv_multipart()
            logger.debug(msg)
            logger.debug('------------------------------------------')
            if not msg[1].decode() == 'END':
                signal = json.loads(msg[1].decode())
                logger.info(signal)
                
                if callback is not None:
                    callback(signal)
        
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM: # Interrupted
                break           
            else:
                logger.error(e)
                raise
        
        except IndexError as e:
            logger.error(e)
            logger.error(msg)
            
        count += 1
        
    end = time.time()
    duration = round(end-start)
    print (
        f"Subscriber received {count} messages in {duration} seconds "\
            f"(={round(count/ duration)} msg/s)"
        )    
    
    
async def main(address, topic):
    ctx = zmq.asyncio.Context()

    try:
        await subscriber(ctx=ctx, address=address, topic=topic)
    except KeyboardInterrupt:
        ctx.term()
        
              
if __name__ == '__main__':
    
    try:
        address = sys.argv[1]
        assert 'tcp://' in address
        assert len(address.split(':')) == 3
        
        topic = '' #sys.argv[2] if len(sys.argv) == 3 else ''
        # assert isinstance(topic, str)
        
        logger.info(f'connecting to {address} and subscribing for topic: {topic}')
        asyncio.run(main(address, topic))
    
    except (IndexError, AssertionError) as e:
        print(e)
        print('usage: python zmq_subscriber.py "tcp://<host ip>:<port>" <topic>')
        print('\nWhen no topic is given, we subscribe to all topics.')
    except KeyboardInterrupt:
        pass
    
        

