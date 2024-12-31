#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 29 18:08:20 2024

@author dhaneor
"""
import asyncio
import logging
import zmq

from .protocol import ROLES, Ready, Hoy, Bye, Task, Result

logger = logging.getLogger(f"main.{__name__}")
logger.setLevel(logging.ERROR)


class Messenger:

    def __init__(
        self,
        origin: str,
        role: ROLES,
        socket: zmq.Socket,
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
            try:
                request, recv_id = await self.queue.get()
            except Exception as e:
                logger.error("[%s] Error processing request: %s" % (self.role, e))
                continue

            if request is None:
                break

            if isinstance(request, str):
                match request:
                    case "HOY":
                        await self.say_hoy(recv_id)
                    case "READY":
                        logger.debug("[%s] Processing READY message..." % self.origin)
                        await self.say_ready(recv_id)
                    case "BYE":
                        await self.say_goodbye(recv_id)
                    case _:
                        logger.error(f"Unknown message type: {request}")

                self.queue.task_done()

            elif isinstance(request, dict):
                match self.role:
                    case ROLES.BROKER:
                        await self.send_task(recv_id, request)
                    case ROLES.WORKER:
                        logger.debug("Queue received result")
                        await self.send_result(recv_id, result=request)
                    case _:
                        logger.error("no action defined for role: %s" % self.role)

                self.queue.task_done()

            else:
                logger.error("Invalid request type: %s" % type(request))
                self.queue.task_done()

        logger.debug("[%s] Messenger stopped." % self.origin)

    async def stop(self):
        """Stop the messenger."""
        await self.queue.put((None, None))

    async def say_hoy(self, recv_id: bytes | None = None):
        """Send a HOY message to a worker."""
        hoy_msg = Hoy(origin=self.origin, role=self.role, recv_id=recv_id)
        logger.debug("[%s] Sending HOY message ..." % self.origin)
        await hoy_msg.send(self.socket)

    async def say_ready(self, recv_id: bytes | None = None):
        """Send a HOY message to a worker."""
        logger.debug("[%s] Sending READY message ..." % self.origin)
        bye_msg = Ready(origin=self.origin, role=self.role, recv_id=recv_id)
        await bye_msg.send(self.socket)

    async def say_goodbye(self, recv_id: bytes | None = None):
        """Send a BYE message."""
        logger.debug("[%s] Sending BYE message ..." % self.origin)
        bye_msg = Bye(origin=self.origin, role=self.role, recv_id=recv_id)
        await bye_msg.send(self.socket)

    async def send_task(self, recv_id: bytes | None = None, task: dict | None = None):
        """Send a TASK message."""
        task_msg = Task(origin=self.origin, role=self.role, recv_id=recv_id, task=task)
        await task_msg.send(self.socket)

    async def send_result(
        self, recv_id: bytes | None = None, result: dict | None = None
    ):
        """Send a RESULT message."""
        result_msg = Result(
            origin=self.origin, role=self.role, recv_id=recv_id, result=result
            )
        try:
            await result_msg.send(self.socket)
        except Exception as e:
            logger.error(f"Error sending result: {e}")
        else:
            logger.debug("[%s] Sent result message." % self.origin)
