#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:50:53 2021

@author: dhaneor
"""
import asyncio
import time
import zmq
from zmq.asyncio import Context
from asyncio import create_task
from statistics import mean as avg


ctx = Context()


async def client():
    client_socket = ctx.socket(zmq.REQ)
    # client_socket.connect("tcp://127.0.0.1:5570")
    client_socket.connect("inproc://queue")
    exc_times = []
    counter = 0
    no_of_messages = 10_000

    print('client: hello')
    # await asyncio.sleep(1)
    print('client: let´s go!')

    st = time.time()

    while counter < no_of_messages:

        if counter % 10 == 0:
            message = str(time.time()).encode()
            await client_socket.send(message)
            message = await client_socket.recv()
            exc_times.append((time.time() - float(message)) * 1_000_000)

        await asyncio.sleep(0.000_3)  # simulate work for 300µs
        counter += 1

    client_socket.send(b'BYE')
    client_socket.close()

    print('client: bye')
    await asyncio.sleep(0.1)

    print(f"average execution time: {avg(exc_times):.2f} microseconds")
    print(f"{no_of_messages} messages in {(time.time() - st):.2f} seconds")
    print(time.time())

    return


async def server():
    server_socket = ctx.socket(zmq.REP)
    # server_socket.bind("tcp://*:{}".format(5570))
    server_socket.bind("inproc://queue")
    print('server: hello')

    await asyncio.sleep(0.1)

    while True:
        message = await server_socket.recv()
        await asyncio.sleep(0.000_01)
        await server_socket.send(message)
        if message == b'BYE':
            break

    print('server: bye')
    server_socket.close()
    return


async def main():
    tasks = [create_task(server()), create_task(client())]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        ctx.term()


if __name__ == '__main__':
    asyncio.run(main())
