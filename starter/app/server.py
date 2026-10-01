#!/usr/bin/env python3
"""Container CMD: one resident ROCm model, private AF_UNIX socket, no TCP port."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import time

# Set these BEFORE importing Hugging Face in any module.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from protocol import atomic_json, receive, send

LOG = logging.getLogger("mc3.server")
REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


class Service:
    def __init__(self, engine, encoder, index_dir: Path):
        self.engine, self.encoder, self.index_dir = engine, encoder, index_dir.resolve()
        self.retriever = None

    def handle(self, request: dict) -> dict:
        operation = request.get("op")
        if operation == "ping":
            return {"ready": True, "indexed": self.retriever is not None}
        if operation == "index":
            from retriever import build_index, Retriever
            directory = Path(request["directory"]).resolve(strict=True)
            if not directory.is_relative_to(self.index_dir):
                raise ValueError("index directory is outside MC3_INDEX_DIR")
            deadline = min(float(request["deadline"]), time.monotonic() + 550)
            if time.monotonic() >= deadline:
                raise TimeoutError("index deadline expired")
            units = json.loads((directory / "units.json").read_text(encoding="utf-8"))
            manifest = build_index(directory, Path(request["corpus"]), units, self.encoder,
                                   generation=request["generation"], deadline=deadline)
            next_retriever = Retriever(directory, self.encoder)
            previous = self.retriever
            self.retriever = next_retriever
            if previous is not None:
                previous.close()
            atomic_json(self.index_dir / "active.json", {"directory": str(directory), **manifest})
            return {"indexed": True, "chunks": manifest["chunks"]}
        if operation == "answer":
            if self.retriever is None:
                raise RuntimeError("no index; run app.py --index first")
            if request.get("corpus") != self.retriever.manifest["corpus"]:
                raise ValueError("query corpus differs from indexed corpus")
            question = request.get("query")
            if not isinstance(question, str) or len(question) > 8000:
                raise ValueError("invalid question")
            deadline = min(float(request["deadline"]), time.monotonic() + 24.0)
            if time.monotonic() >= deadline - 0.5:
                return dict(REFUSAL)
            chunks = self.retriever.retrieve(question, deadline=deadline)
            if not chunks:
                return dict(REFUSAL)
            return self.engine.answer(question, chunks, deadline=deadline)
        raise ValueError("unknown operation")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index_dir = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
    socket_path = Path(os.environ.get("MC3_SOCKET", "/tmp/mc3.sock"))
    from inference import AnswerEngine
    from retriever import DenseEncoder, Retriever
    # GPU model is substantive computation, not a dummy memory reservation.
    engine = AnswerEngine(os.environ.get("MC3_LLM_MODEL", "/models/qwen"))
    encoder = DenseEncoder(os.environ.get("MC3_EMBED_MODEL", "/models/embedding"))
    engine.answer("What is the code?", [{"id": 0, "path": "warmup.txt", "location": "1",
                  "text": "The code is READY-1.", "score": 1.0}], deadline=time.monotonic() + 45)
    encoder.encode(["warmup"])
    service = Service(engine, encoder, index_dir)
    active = index_dir / "active.json"
    if active.exists():
        try:
            stored = json.loads(active.read_text(encoding="utf-8"))
            service.retriever = Retriever(Path(stored["directory"]), encoder)
        except Exception:
            LOG.exception("Prior index unavailable; waiting for --index")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # Do not steal a socket from another live server.
    if socket_path.exists():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            try:
                probe.connect(str(socket_path))
            except (ConnectionRefusedError, FileNotFoundError):
                socket_path.unlink(missing_ok=True)
            else:
                raise RuntimeError("another MC3 server is already running")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        socket_path.chmod(0o600)
        listener.listen(8)
        listener.settimeout(0.5)
        LOG.info("GPU warm; listening on %s", socket_path)
        try:
            while not stopping:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    try:
                        request = receive(connection, time.monotonic() + 2)
                        result = service.handle(request)
                    except Exception as exc:
                        LOG.exception("Request failed")
                        result = {**REFUSAL, "error": str(exc)[:500]}
                    try:
                        connection.settimeout(0.5)
                        send(connection, result)
                    except OSError:
                        LOG.warning("Client disconnected before result was ready")
        finally:
            socket_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
