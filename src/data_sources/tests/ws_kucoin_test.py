#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:44:23 2023

@author_ dhaneor
"""
import asyncio
import sys

from os.path import dirname as dir

sys.path.append(dir(dir(dir(__file__))))

import data_sources.websockets.ws_kucoin as ws  # noqa: E402


# --------------------------------------------------------------------------------------
async def test_connection_prep_topic():
    c = ws.Connection(None, True)

    topics = ["topic"] * 50

    topics, too_much = await c._prep_topic_str(topics)

    print(topics)
    print(too_much)


if __name__ == "__main__":
    asyncio.run(test_connection_prep_topic())
