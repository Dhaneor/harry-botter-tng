#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:50:53 2021

@author: dhaneor
"""
import asyncio
import zmq


async def load_balancer(container_port, worker_port, zmq_context=None):
    # Initialize the ZeroMQ context if not provided
    context = zmq_context or zmq.Context()

    # Set up the PULL socket to get data from the OHLCV Container
    container_socket = context.socket(zmq.PULL)
    container_socket.bind(f"tcp://*:{container_port}")

    # Set up the PUSH socket to send tasks to the Strategy Workers
    worker_socket = context.socket(zmq.PUSH)
    worker_socket.bind(f"tcp://*:{worker_port}")

    while True:
        # Wait for data (Container Class) from the OHLCV Container
        container_instance = await container_socket.recv_pyobj()

        # Distribute the container instance to the next available Strategy Worker
        await worker_socket.send_pyobj(container_instance)

# Sample port numbers, can be changed as required
# Uncomment the next line to start the load balancer (infinite loop)
# asyncio.run(load_balancer(5555, 5556))
