#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 23:54:28 2021

@author: dhaneor
"""
import requests
from threading import Thread
from multiprocessing import Queue
from datetime import datetime
import time

from broker.models.account import Account

# =============================================================================
class Thoth(Thread):

    def __init__(self, event_queue: Queue):

        Thread.__init__(self, daemon=True)

        self.events_queue = event_queue
        self.telegram = Telegram()
        self.keep_going = True
        self.start()

    # ------------------------------------------------------------------------- 
    def run(self):
        while True:
            if not self.events_queue.empty():
                self._process_event(self.events_queue.get())
            else: 
                time.sleep(1)        

    # -------------------------------------------------------------------------
    def _process_event(self, event):
        if isinstance(event, str):
            self._send_telegram_message(event)

    def _send_telegram_message(self, message):
        self.telegram.send(message)
        
        
# =============================================================================
class Telegram:

    def __init__(self, chat_id=None):

        self.bot_token = '1406970784:AAGB7t29hz9DR2X8SMtYFfvet2GcKTNO12w'
        
        if chat_id is None: self.bot_chatID = '600751557'
        else: self.bot_chatID = chat_id

    # -------------------------------------------------------------------------
    def run(self):

        while True:

            time.sleep(1)


    def send(self, message):
        
        send_text = 'https://api.telegram.org/bot' 
        send_text += self.bot_token + '/sendMessage?chat_id=' + self.bot_chatID 
        send_text += f'&parse_mode=Markdown&text={message}'

        try:
            response = requests.get(send_text)
        except Exception:
            pass

        return
        

# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':

    q = Queue()

    thoth = Thoth(event_queue=q)
    # thoth.start()

    print('[MAIN] puttin something in the queue ...')

    text = "*ADAUSDT* balance update\n"
    text += "15.03453 ADA\n"
    text += "23897.8473 USDT"

    q.put(text)
    time.sleep(1)
    q.put('two')
    time.sleep(1)
    q.put('three')
    time.sleep(1)
