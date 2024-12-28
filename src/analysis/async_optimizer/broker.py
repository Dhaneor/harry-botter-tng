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

# Configure logging
logger = logging.getLogger("main.broker")
logger.setLevel(logging.INFO)


async def broker(task_list, worker_timeout=5):
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
    broker_socket.bind("tcp://*:5555")

    logger.info("[BROKER] Started and waiting for workers...")

    # Keep track of ready workers
    ready_workers = []
    shutdown_initiated = False

    await asyncio.sleep(2)  # Wait for workers to connect

    try:
        while True:  # task_list or ready_workers or alive:
            if not ready_workers and shutdown_initiated:
                break

            try:
                # Wait for incoming worker messages with timeout
                message = await asyncio.wait_for(
                    broker_socket.recv_multipart(), timeout=worker_timeout
                )

                logger.debug("[BROKER] Received message from worker %s" % message)

                if message:
                    logger.debug(message)
                    hex_id, worker_id, _, msg = message
                    worker_id = worker_id.decode()
                    logger.debug("worker %s: %s" % (worker_id, msg.decode()))

                    if msg == b"READY":
                        ready_workers.append(worker_id)
                        logger.info(
                            "[BROKER] Worker %s is ready. [%s]"
                            % (worker_id,  len(ready_workers))
                            )

                    elif msg == b"BYE":
                        if worker_id in ready_workers:
                            ready_workers.remove(worker_id)
                        logger.info(
                            "[BROKER] Worker %s is leaving ... [%s]"
                            % (worker_id, len(ready_workers))
                        )

                    else:
                        logger.info(
                            "[BROKER] Unexpected message from worker %s: %s",
                            worker_id,
                            msg,
                        )

            except asyncio.TimeoutError:
                logger.info("[BROKER] Timeout while waiting for worker messages.")
                break

            # Assign tasks if workers are ready and tasks remain
            while task_list and ready_workers:
                worker_id = ready_workers.pop(0)
                task = task_list.pop(0)
                logger.info(
                    "[BROKER] Sending task '%s' to worker %s", task, worker_id
                )
                await broker_socket.send_multipart([hex_id, b"", task.encode()])

            if not (task_list and shutdown_initiated):
                for worker_id in ready_workers:
                    await broker_socket.send_multipart([hex_id, b"", "DONE".encode()])
                    logger.info("[BROKER] Sent 'DONE' to worker %s", worker_id)

    except KeyboardInterrupt:
        logger.info("[BROKER] Interrupted by user. Shutting down...")

    finally:
        logger.error("ready workers: %s" % len(ready_workers))

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
