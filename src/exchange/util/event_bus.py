#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 13:30:53 2022

@author: dhaneor
"""
import logging
from typing import Union, List
from abc import ABC  

logger = logging.getLogger('main.event_bus')
logger.setLevel(logging.WARNING)


# =============================================================================
class Event(ABC):
    
    def __init__(self):
         self.name: str = ''
         self.data = None
         
class OrderUpdatedEvent(Event):
    
    def __init__(self):
        super().__init__()
                 
class OrderCreatedEvent(OrderUpdatedEvent):
    
    def __init__(self, order:object):
        super().__init__()
        self.data = order
      
class OrderFilledEvent(OrderUpdatedEvent):
    
    def __init__(self, order:object):
        super().__init__()
        self.data = order

    
class OrderCancelledEvent(OrderUpdatedEvent):
    
    def __init__(self, order_id:Union[str, int]):
        super().__init__()
        self.data = order_id

class LoanCreatedEvent(Event):
    
    def __init__(self, details):
        super().__init__()
        self.data = details

class LoanRepaidEvent(Event):
    
    def __init__(self, details):
        super().__init__()
        self.data = details
        

        
# ==============================================================================
class EventBus:
    
    def __init__(self):
        
        self._subscribers: dict = {}
    
    # --------------------------------------------------------------------------
    def register_subscriber(self, subscriber:object, event_types:List[Event]):
        self._subscribers[subscriber] = event_types
        logger.debug(f'successfully registered <{subscriber}> ({event_types})')
        
    def unregister_subscriber(self, subscriber:object):    
        try:
            del self._subscribers[subscriber]
        except Exception as e:
            logger.debug(f'[EventBus] unable to unregister {subscriber}')

    # --------------------------------------------------------------------------
    def publish_event(self, event: Event):

        for subscriber, event_types in self._subscribers.items():
            if isinstance(event, tuple(event_types)):
                logger.debug(f'notifying {subscriber} about {event}')
                subscriber.handle_event(event)



    

    
         