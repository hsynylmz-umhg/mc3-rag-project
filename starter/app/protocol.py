"""Bounded JSON-over-Unix-socket protocol; no third-party imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time

MAX_MESSAGE = 131072


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".mc3-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_exact(connection: socket.socket, length: int, deadline: float) -> bytes:
    chunks = []
    while length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("IPC deadline expired")
        connection.settimeout(remaining)
        piece = connection.recv(length)
        if not piece:
            raise ConnectionError("server closed the connection")
        chunks.append(piece)
        length -= len(piece)
    return b"".join(chunks)


def receive(connection: socket.socket, deadline: float) -> dict:
    length = struct.unpack("!I", _read_exact(connection, 4, deadline))[0]
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("invalid IPC frame size")
    value = json.loads(_read_exact(connection, length, deadline))
    if not isinstance(value, dict):
        raise ValueError("IPC message must be an object")
    return value


def send(connection: socket.socket, payload: dict) -> None:
    message = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(message) > MAX_MESSAGE:
        raise ValueError("IPC message too large")
    connection.sendall(struct.pack("!I", len(message)) + message)


def rpc(path: str, payload: dict, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(path)
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        send(connection, payload)
        return receive(connection, deadline)
