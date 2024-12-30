#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 28 03:46:20 2024

@author dhaneor
"""

import asyncio
import zmq
import zmq.asyncio
import logging

from .messenger import Messenger
from .protocol import TYPE, ROLES, Message, Hoy, Bye, Task

# Configure logging
logger = logging.getLogger("main.broker")
logger.setLevel(logging.INFO)

NAME = "BROKER"
ROLE = ROLES.BROKER


async def say_hoy(recv_id: bytes, socket: zmq.asyncio.Socket):
    """Send a HOY message to a worker."""
    bye_msg = Hoy(recv_id=recv_id)
    await bye_msg.send(socket)


async def say_goodbye(recv_id: bytes, socket: zmq.asyncio.Socket):
    """Send a BYE message."""
    bye_msg = Bye(recv_id=recv_id)
    await bye_msg.send(socket)


async def broker(
    ctx: zmq.asyncio.Context | None = None,
    task_list: list | None = None,
    addr: str | None = None,
    worker_timeout: int = 5,
):
    """
    Async ZeroMQ Broker for distributing backtesting tasks to workers.

    Args:
        task_list (list): List of tasks to distribute. Each task is a string.
        worker_timeout (int): Time in seconds to wait for workers to report ready.

    Workflow:
        - Workers send 'READY' when they're available.
        - Broker assigns tasks from task_list.
        - If no tasks remain, the broker sends 'DONE' to workers.
        - Workers send 'BYE' when they're leaving.
    """

    context = ctx or zmq.asyncio.Context()
    broker_socket = context.socket(zmq.ROUTER)
    broker_socket.bind(addr)

    poller = zmq.asyncio.Poller()
    poller.register(broker_socket, zmq.POLLIN)

    logger.debug("[BROKER] Started and waiting for workers at %s ..." % addr)

    q = asyncio.Queue()
    messenger = Messenger(origin=NAME, role=ROLE, socket=broker_socket, queue=q)

    # Keep track of ready workers
    known_workers = set()
    ready_workers = []
    shutdown_initiated = False

    # await asyncio.sleep(worker_timeout)  # Simulate worker startup delay

    try:
        while True:  # task_list or ready_workers or alive:
            try:
                events = await poller.poll()
                for socket, _ in events:
                    if socket == broker_socket:
                        multipart = await broker_socket.recv_multipart()
                        msg = Message.from_multipart(multipart)

                logger.debug(
                    "[%s] Received %a message from worker %s"
                    % (NAME, msg.type.name, msg.origin)
                )

                if msg:
                    worker = (msg.origin, msg.recv_id)

                    if msg.type == TYPE.READY:
                        if shutdown_initiated:
                            await messenger.say_goodbye(msg.recv_id)
                        else:
                            ready_workers.append(worker)
                            logger.debug(
                                "[BROKER] %s is ready. [ready: %s]"
                                % (worker[0], len(ready_workers))
                            )

                    elif msg.type == TYPE.HOY:
                        known_workers.add(worker)
                        ready_workers.append(worker)
                        logger.debug(
                            "[BROKER] 'HOY' from %s ... [known: %s]"
                            % (worker[0], len(known_workers))
                        )

                    elif msg.type == TYPE.BYE:
                        if worker in ready_workers:
                            ready_workers.remove(worker)
                        if worker in known_workers:
                            known_workers.remove(worker)
                            logger.debug(
                                "[BROKER] 'BYE' from %s ... [known: %s]"
                                % (worker[0], len(known_workers))
                            )
                    else:
                        logger.info(
                            "[BROKER] Unexpected message type from worker %s: %s",
                            worker[0],
                            msg.type,
                        )

            except asyncio.TimeoutError:
                logger.warning("[BROKER] Timeout while waiting for worker messages.")
                break

            # Assign tasks if workers are ready and tasks remain
            while task_list and ready_workers:
                name, recv_id = ready_workers.pop(0)
                task = task_list.pop(0)
                logger.debug("[BROKER] Sending task '%s' to worker %s", task, name)
                reply = Task(origin=NAME, role=ROLE, recv_id=recv_id, task=task)
                await reply.send(broker_socket)

            logger.debug(
                "task list: %s, ready workers: %s, known workers: %s",
                len(task_list),
                len(ready_workers),
                len(known_workers),
            )
            logger.debug("shutdown intiated: %s" % shutdown_initiated)

            if not task_list:
                if not shutdown_initiated:
                    for worker in ready_workers:
                        await q.put_nowait(("BYE", worker[1]))
                        logger.info("[BROKER] Sent 'BYE' to worker %s", name)
                    shutdown_initiated = True

            if not known_workers:
                if task_list:
                    logger.info("[BROKER] No workers available. Waiting ...")
                else:
                    logger.info(
                        "[%s] Work completed, workers gone. Shutting down..." % NAME
                    )
                    break

    except KeyboardInterrupt:
        logger.info("[BROKER] Interrupted by user. Shutting down...")

    finally:
        await messenger.stop()
        broker_socket.close()
        context.term()
        logger.info("[BROKER] Shutdown complete.")


# Example usage
if __name__ == "__main__":
    tasks = [
        "Task 1",
        "Task 2",
        "Task 3",
        "Task 4",
        "Task 5",
    ]

    asyncio.run(broker(tasks))
