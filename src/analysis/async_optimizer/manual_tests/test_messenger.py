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
    MT, ROLES, Message, Bye, Task, Hoy
)

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

        if msg.type == MT.BYE:
            break

    print("Client shutting down...")
    socket.close()


async def worker(context):
    socket = context.socket(zmq.DEALER)
    socket.connect(CLIENT_ADDRESS)

    client = Messenger(origin="worker", role=ROLES.WORKER, socket=socket)

    await client.say_hoy()
    await client.say_goodbye()
    socket.close()


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
