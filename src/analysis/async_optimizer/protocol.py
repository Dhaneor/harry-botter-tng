#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 29 01:59:20 2024

@author dhaneor
"""

import asyncio
import json
import logging
import time
import zmq
from enum import Enum
from pprint import pprint

logger = logging.getLogger(f"main.{__name__}")


class TYPE(Enum):
    HOY = "HOY"
    READY = "READY"
    BYE = "BYE"
    TASK = "TASK"
    RESULT = "RESULT"
    ERROR = "ERROR"


class ROLES(Enum):
    CLIENT = "CLIENT"
    BROKER = "BROKER"
    WORKER = "WORKER"
    COLLECTOR = "COLLECTOR"


class Message:
    def __init__(
        self,
        type: TYPE,
        origin: str,
        role: ROLES,
        socket: zmq.Socket | None = None,
        recv_id: bytes | None = None,
        payload: str | None = None,
        payload_bytes: bytes | None = None,
    ):
        self.type = type
        self.origin = origin
        self.role = role
        self.socket = socket
        self.recv_id = recv_id
        self.payload = payload
        self.payload_bytes = payload_bytes

    @staticmethod
    def from_multipart(parts):
        """
        Parse a multipart ZeroMQ message into a Message object.

        Expected Formats:
        - ROUTER Socket:
        [recv_id, type, origin, payload (optional), payload_bytes (optional)]
        - Non-ROUTER Socket:
        [type, origin, payload (optional), payload_bytes (optional)]
        """
        if len(parts) < 2:
            raise ValueError("Invalid message format: Not enough parts.")

        # Check if it's a ROUTER socket message (recv_id is present)
        if len(parts) > 5:
            recv_id = parts.pop(0)  # First part is recv_id for ROUTER
        else:
            recv_id = None

        # Extract mandatory parts
        try:
            msg_type = TYPE(parts[0].decode())  # First part is message type
            origin = parts[1].decode()  # Second part is origin
            role = ROLES(parts[2].decode())  # Third part is role
        except (IndexError, UnicodeDecodeError, ValueError) as e:
            logger.error(parts[0], parts[1], parts[2], e)
            raise ValueError(f"Failed to parse type or origin from message: {e}")

        # Extract optional payload
        payload = None
        if len(parts) > 2 and parts[3]:
            try:
                payload = json.loads(parts[3].decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = parts[3].decode(errors="replace")

        # Extract optional payload_bytes
        payload_bytes = parts[4] if len(parts) > 4 else None

        return Message(
            type=msg_type,
            origin=origin,
            role=role,
            recv_id=recv_id,
            payload=payload,
            payload_bytes=payload_bytes,
        )

    async def get_multipart(self):
        return [
            self.type.name.encode(),
            self.origin.encode(),
            self.role.name.encode() if self.role else b"",
            self._encode_payload(),
            self.payload_bytes if self.payload_bytes else b"",
        ]

    async def send(self, socket):
        message_parts = await self.get_multipart()
        socket = socket or self.socket

        # Add recv_id for ROUTER sockets
        if socket.type == zmq.ROUTER:
            if self.recv_id is None:
                raise ValueError("recv_id is required for ROUTER sockets")
            message_parts.insert(0, self.recv_id)

        try:
            await socket.send_multipart(message_parts)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def _encode_payload(self):
        if self.payload is None:
            return b""

        if isinstance(self.payload, (list, dict)):
            return json.dumps(self.payload).encode()
        elif isinstance(self.payload, (int, float)):
            return str(self.payload).encode()
        elif isinstance(self.payload, str):
            return self.payload.encode()
        else:
            raise ValueError(f"Unsupported payload type: {type(self.payload)}")


class Hoy(Message):
    def __init__(self, origin: str, role: ROLES, recv_id: bytes | None = None):
        super().__init__(type=TYPE.HOY, origin=origin, role=role, recv_id=recv_id)


class Ready(Message):
    def __init__(self, origin: str, role: ROLES, recv_id: bytes | None = None):
        super().__init__(type=TYPE.READY, origin=origin, role=role, recv_id=recv_id)


class Bye(Message):
    def __init__(self, origin: str, role: ROLES, recv_id: bytes | None = None):
        super().__init__(type=TYPE.BYE, origin=origin, role=role, recv_id=recv_id)


class Task(Message):
    def __init__(
        self,
        origin: str,
        role: ROLES,
        recv_id: bytes | None = None,
        task: dict | None = None
    ):
        super().__init__(
            type=TYPE.TASK, origin=origin, role=role, recv_id=recv_id, payload=task
            )


class Result(Message):
    def __init__(
        self,
        origin: str,
        role: ROLES,
        recv_id: bytes | None = None,
        result: dict | None = None
    ):
        super().__init__(
            type=TYPE.RESULT, origin=origin, role=role, recv_id=recv_id, payload=result
            )


class Error(Message):
    def __init__(
        self,
        origin: str,
        role: ROLES,
        recv_id: bytes | None = None,
        error: dict | None = None
    ):
        super().__init__(
            type=TYPE.READY, origin=origin, role=role, recv_id=recv_id, payload=error
            )


# ================================ Some simple tests =================================
async def test_hoy():
    st = time.time()
    for _ in range(1_000_000):
        msg = Hoy("client_1", ROLES.WORKER)
    print(f"Hoy time: {(time.time() - st):.2f} µs")
    pprint(msg.__dict__)


async def test_ready():
    st = time.time()
    for _ in range(1_000_000):
        msg = Ready("client_1", ROLES.WORKER)
    print(f"Ready time: {(time.time() - st):.2f} ��s")
    print(await msg.get_multipart())
    pprint(msg.__dict__)


async def test_task():
    print("-" * 80)
    st = time.time()
    for _ in range(1_000_000):
        msg = Task("client_1", ROLES.WORKER, task={"key": "value"})
    print(f"Ready time: {(time.time() - st):.2f} ��s")
    pprint(msg.__dict__)
    print("-" * 80)

    parts = await msg.get_multipart()
    print(parts)
    try:
        msg = Message.from_multipart(parts)
    except Exception as e:
        print(f"Error parsing message: {e}")
        return

    print(msg.__dict__)
    print(msg.type == TYPE.TASK)

    assert await msg.get_multipart() == parts


async def main():
    tasks = [
        test_hoy(),
        test_ready(),
        test_task(),
        # Add more test cases here...
    ]
    asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

    multipart = [TYPE.READY, b"client_1", ROLES.WORKER, b'{"message": "Hello, Hoy!"}']

    print(ROLES.WORKER.name)
