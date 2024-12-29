#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 29 18:08:20 2024

@author dhaneor
"""
import asyncio
import logging
import zmq

from .protocol import MT, ROLES, Message, Ready, Hoy, Bye, Task  # noqa: F401

logger = logging.getLogger("main.messenger")


class Messenger:

    def __init__(
        self,
        origin: str,
        role: ROLES,
        socket: zmq.Socket | None = None,
        queue: asyncio.Queue | None = None
    ):
        if socket is None and queue is None:
            raise ValueError("Either socket or queue must be provided")
        self.origin = origin
        self.role = role
        self.socket = socket
        self.queue = queue

    async def run(self):
        """
        Main worker loop.

        Sends messages that clients have put into the queue.
        """
        while True:
            task = await self.queue.get()
            self.socket, message = task

            if message is None:
                logger.info("Received shutdown request...")
                break

            if isinstance(message, list):
                try:
                    logger.debug("sending results to oracle...")
                    await self.socket.send_multipart(message)
                    self.queue.task_done()
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    self.queue.task_done()
            elif isinstance(message, dict):
                try:
                    await self.socket.send_json(message)
                    self.queue.task_done()
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    self.queue.task_done()

    async def say_hoy(self, recv_id: bytes | None = None):
        """Send a HOY message to a worker."""
        hoy_msg = Hoy(origin=self.origin, role=self.role, recv_id=recv_id)
        logger.debug("Sending HOY message: %s", hoy_msg)
        await hoy_msg.send(self.socket)

    async def say_ready(self, recv_id: bytes | None = None):
        """Send a HOY message to a worker."""
        bye_msg = Hoy(recv_id=recv_id)
        await bye_msg.send(self.socket)

    async def say_goodbye(self, recv_id: bytes | None = None):
        """Send a BYE message."""
        bye_msg = Bye(origin=self.origin, role=self.role, recv_id=recv_id)
        await bye_msg.send(self.socket)