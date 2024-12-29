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

from .protocol import MT, ROLES, Message, Hoy, Bye, Task

# Configure logging
logger = logging.getLogger("main.broker")
logger.setLevel(logging.INFO)

NAME = "BROKER"


async def process_message(socket, message):
    ...


async def say_hoy(recv_id: bytes, socket: zmq.asyncio.Socket):
    """Send a HOY message to a worker."""
    bye_msg = Hoy(recv_id=recv_id)
    await bye_msg.send(socket)


async def say_goodbye(recv_id: bytes, socket: zmq.asyncio.Socket):
    """Send a BYE message."""
    bye_msg = Bye(recv_id=recv_id)
    await bye_msg.send(socket)


async def broker(task_list, addr: str, worker_timeout=5):
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

    context = zmq.asyncio.Context()
    broker_socket = context.socket(zmq.ROUTER)
    broker_socket.bind(addr)

    poller = zmq.asyncio.Poller()
    poller.register(broker_socket, zmq.POLLIN)

    logger.info("[BROKER] Started and waiting for workers at %s ..." % addr)

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
                        message = Message.from_multipart(multipart)

                # logger.debug("[BROKER] Received message from worker %s" % message)

                if message:
                    hex_id, worker_id, _, msg = message
                    worker_id = worker_id.decode()

                    if msg.type == MT.READY:
                        if shutdown_initiated:
                            await say_goodbye(msg.recv_id, broker_socket)
                        else:
                            ready_workers.append(msg.origin)
                            logger.debug(
                                "[BROKER] %s is ready. [ready: %s]"
                                % (worker_id,  len(ready_workers))
                                )

                    elif msg.type == MT.HOY:
                        known_workers.add(msg.origin)
                        ready_workers.append(msg.origin)
                        logger.info(
                            "[BROKER] 'HOY' from %s ... [known: %s]"
                            % (worker_id, len(known_workers))
                            )

                    elif msg == b"BYE":
                        if worker_id in ready_workers:
                            ready_workers.remove(worker_id)
                        if worker_id in known_workers:
                            known_workers.remove(worker_id)
                            logger.debug(
                                "[BROKER] 'BYE' from %s ... [known: %s]"
                                % (worker_id, len(known_workers))
                            )
                    else:
                        logger.info(
                            "[BROKER] Unexpected message from worker %s: %s",
                            worker_id,
                            msg,
                        )

            except asyncio.TimeoutError:
                logger.warning("[BROKER] Timeout while waiting for worker messages.")
                break

            # Assign tasks if workers are ready and tasks remain
            while task_list and ready_workers:
                worker_id = ready_workers.pop(0)
                task = task_list.pop(0)
                logger.debug(
                    "[BROKER] Sending task '%s' to worker %s", task, worker_id
                )
                await broker_socket.send_multipart([hex_id, b"", task.encode()])

            logger.debug(
                "task list: %s, ready workers: %s, known workers: %s",
                len(task_list), len(ready_workers), len(known_workers)
                )
            logger.debug("shutdown intiated: %s" % shutdown_initiated)

            if not task_list:
                if not shutdown_initiated:
                    for worker_id in ready_workers:
                        await broker_socket.send_multipart([hex_id, b"", "DONE".encode()])
                        logger.info("[BROKER] Sent 'DONE' to worker %s", worker_id)
                    shutdown_initiated = True

            if not known_workers:
                if task_list:
                    logger.info("[BROKER] No workers available. Waiting ...")
                else:
                    logger.info(
                        "[BROKER] Work completed, all workers are gone. Shutting down..."
                        )
                    break

    except KeyboardInterrupt:
        logger.info("[BROKER] Interrupted by user. Shutting down...")

    finally:
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
