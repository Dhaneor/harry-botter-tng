#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nonv 28 14:09:23 2021

@author_ dhaneor
"""
import sys
import os
import asyncio
import zmq
import zmq.asyncio
import logging
import queue
import logging
import datetime as dt

from logging.handlers import QueueHandler, QueueListener
from random import choice

que = queue.Queue(-1)  # no limit on size
queue_handler = QueueHandler(que)
handler = logging.StreamHandler()
listener = QueueListener(que, handler)
root = logging.getLogger('main')
root.addHandler(queue_handler)
root.addHandler(handler)
formatter = logging.Formatter('%(threadName)s: %(message)s')
handler.setFormatter(formatter)
listener.start()

# ------------------------------------------------------------------------------
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from data_sources.util.publishers import (IPublisher, PrintPublisher, LogPublisher,
                                      EventBusPublisher, ZeroMqPublisher)
from src.infrastructure.event_bus import EventBus
from src.helpers.timeops import execution_time

messages = [
    'one message', 'another message', 'really important message',
    'happy message', 'heavenly message', 'message from pleiades'
]



# =============================================================================
@execution_time
async def main():
    # eb = EventBus()
    # p = EventBusPublisher(event_bus=eb)
    
    # ctx = zmq.asyncio.Context()
    p = ZeroMqPublisher()
    
    while True:
        try:
            msg = choice(messages)
            print(f'{dt.datetime.utcnow()}: sending message to publisher: {msg}')
            await p.publish(msg)
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            p.ctx.destroy()
            await asyncio.sleep(2)
            break

    # coros = [p.publish(choice(messages)) for _ in range(5)]
    # await asyncio.gather(*coros)
    
    root.critical('we did it!')

# =============================================================================    
if __name__ == '__main__':
    asyncio.run(main())
    listener.stop()
