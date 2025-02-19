#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import asyncio

from analysis.telegram_signal import create_signal
from util import get_logger

logger = get_logger("main")

# ....................................................................................
positions = [
    # open long position
    {
        "symbol": "BTC/USDT",
        "position_type": "LONG",
        "required_action": "buy",
        "change": "open",
        "change_percent": 5.0,
        "leverage": 0.7389603775496516,
        "pnl_percentage": None,
        "max_drawdown": None,
        "duration": 0,
        "entry_price": 67074.14,
        "entry_time": "15.10. 00:00:00",
        "current_price": 67074.14,
        "exit_price": None,
        "is_open": True,
        "is_new": True,
    },
    # close long position
    {
        "symbol": "BTC/USDT",
        "position_type": "LONG",
        "required_action": "sell",
        "change": "close",
        "change_percent": 5.0,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price": 67074.14,
        "entry_time": "15.10. 00:00:00",
        "current_price": 90333.0,
        "exit_price": 90333.0,
        "is_open": False,
        "is_new": False,
    },
    # open short position
    {
        "symbol": "BTC/USDT",
        "position_type": "SHORT",
        "required_action": "sell",
        "change": "open",
        "change_percent": 5.0,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price": 91333,
        "entry_time": "15.10. 00:00:00",
        "current_price": 91333.0,
        "exit_price": None,
        "is_open": True,
        "is_new": True,
    },
    # close short position
    {
        "symbol": "BTC/USDT",
        "position_type": "SHORT",
        "required_action": "buy",
        "change": "close",
        "change_percent": 5.0,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price": 91333.0,
        "entry_time": "15.10. 00:00:00",
        "current_price": 67074.14,
        "exit_price": 67074.14,
        "is_open": False,
        "is_new": False,
    },
    # hold position
    {
        "symbol": "BTC/USDT",
        "position_type": "LONG",
        "required_action": None,
        "change": None,
        "change_percent": None,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price":  67047.14,
        "entry_time": "15.10. 00:00:00",
        "current_price": 91333.0,
        "exit_price": None,
        "is_open": True,
        "is_new": False,
    },
    # increase long position
    {
        "symbol": "BTC/USDT",
        "position_type": "LONG",
        "required_action": "BUY",
        "change": "increase",
        "change_percent": 5.2348,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price": 67047.14,
        "entry_time": "15.10. 00:00:00",
        "current_price": 96333.0,
        "exit_price": None,
        "is_open": True,
        "is_new": False,
    },
    # decrease long position
    {
        "symbol": "BTC/USDT",
        "position_type": "LONG",
        "required_action": "SELL",
        "change": "decrease",
        "change_percent": 5.0,
        "leverage": 0.7389603775496516,
        "pnl_percentage": 32.29536834653604,
        "max_drawdown": 0.06717968797474501,
        "duration": 123_867,
        "entry_price": 67047.14,
        "entry_time": "15.10. 00:00:00",
        "current_price": 96423.45,
        "exit_price": None,
        "is_open": True,
        "is_new": False,
    },
]

chat_id = "-1002318654276"


# ....................................................................................
async def test_create_signal(position: dict):
    print(await create_signal(position))


async def main():
    for p in positions:
        print("-" * 120)
        print(p)
        await test_create_signal(p)



if __name__ == "__main__":
    asyncio.run(main())
