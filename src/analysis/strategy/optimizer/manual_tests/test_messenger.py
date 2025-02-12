#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 29 18:08:20 2024

@author dhaneor
"""
import asyncio
import zmq
import zmq.asyncio
from asyncio import create_task

from analysis.async_optimizer.messenger import Messenger
from analysis.async_optimizer.protocol import (  # noqa: F401
    TYPE, ROLES, Message, Bye, Task, Hoy
)
from util import get_logger

logger = get_logger("main")

MESSENGER_ADDRESS = "tcp://localhost:5555"
CLIENT_ADDRESS = "tcp://localhost:5556"


async def client(context):
    """
    Client-side worker.

    Connects to a broker, sends a task, and waits for the result.
    """
    socket = context.socket(zmq.ROUTER)
    socket.bind(CLIENT_ADDRESS)

    while True:
        msg = Message.from_multipart(await socket.recv_multipart())
        print(f"Received message from broker: {await msg.get_multipart()}")

        if msg.type == TYPE.BYE:
            break

    print("Client shutting down...")
    socket.close()


async def worker(context):
    socket = context.socket(zmq.DEALER)
    socket.connect(CLIENT_ADDRESS)

    q = asyncio.Queue()
    client = Messenger(queue=q, origin="broker", role=ROLES.BROKER, socket=socket)

    await client.say_hoy()
    await client.say_ready()
    await client.send_task({"name": "example task"})
    await client.send_result({"result": "example result"})
    await client.say_goodbye()

    await q.put(None)  # signal shutdown
    socket.close()
    await asyncio.sleep(0.1)  # wait for worker to shutdown


async def main():
    """Main entry point."""
    context = zmq.asyncio.Context()
    tasks = [create_task(worker(context)), create_task(client(context))]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        context.term()


if __name__ == "__main__":
    asyncio.run(main())
