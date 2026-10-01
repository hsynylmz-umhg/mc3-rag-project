#!/usr/bin/env python3
"""MC-3 CLI. Query invocations import only the standard library and protocol."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import time
import uuid

from protocol import atomic_json, rpc

OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", os.environ.get("MC2_OUTPUT_DIR", "/app/output")))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
SOCKET_PATH = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
LOG = logging.getLogger("mc3")


def index(corpus: Path) -> None:
    """Parse once, then ask the already resident server to persist the index."""
    started = time.monotonic()
    deadline = started + min(float(os.environ.get("MC3_STARTUP_SECONDS", "560")), 570.0)
    corpus = corpus.resolve(strict=True)
    if not corpus.is_dir():
        raise ValueError("corpus must be a directory")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    destination = INDEX_DIR / ("generation-" + generation)
    destination.mkdir()
    from parser import parse_corpus
    units, errors = parse_corpus(corpus, total_timeout=210.0)
    atomic_json(destination / "units.json", units)
    atomic_json(destination / "parse_errors.json", errors)
    LOG.info("Parsed %d units; %d skips/warnings", len(units), len(errors))
    # Parsing overlaps the model load performed by the container CMD.
    while time.monotonic() < deadline:
        try:
            status = rpc(SOCKET_PATH, {"op": "ping"}, timeout=0.5)
            if status.get("ready"):
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    else:
        raise TimeoutError("resident server did not become ready during startup")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("startup budget exhausted")
    result = rpc(SOCKET_PATH, {"op": "index", "corpus": str(corpus),
                 "directory": str(destination.resolve()), "generation": generation,
                 "deadline": deadline}, timeout=remaining)
    if not result.get("indexed"):
        raise RuntimeError(result.get("error", "index build failed"))
    LOG.info("Index ready in %.1fs (%d chunks)", time.monotonic() - started, result["chunks"])


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """IPC only. Never load a model, reparse a corpus, or retry on timeout."""
    try:
        result = rpc(SOCKET_PATH, {"op": "answer", "corpus": str(corpus.resolve()),
                     "query": query, "deadline": time.monotonic() + 25.0}, timeout=26.0)
        value, cites = result.get("answer"), result.get("citations")
        if not isinstance(value, str) or not isinstance(cites, list):
            raise ValueError("invalid worker response")
        if not value:
            return "", [], 0.0
        if not cites or any(not isinstance(p, str) or not p or "\\" in p or p.startswith("/")
                            or ":" in p or ".." in p.split("/") for p in cites):
            raise ValueError("invalid worker citations")
        confidence = float(result.get("confidence", 0.0))
        if not math.isfinite(confidence):
            confidence = 0.0
        return value, sorted(set(cites)), min(1.0, max(0.0, confidence))
    except Exception:
        LOG.exception("Query failed; writing a well-formed empty result")
        return "", [], 0.0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args(argv)
    if args.index is not None:
        if args.corpus is not None or args.query_id is not None or args.query is not None:
            ap.error("--index cannot be combined with query flags")
        index(args.index)
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", args.query_id):
        ap.error("query-id must be a safe filename stem")
    # Create a valid fallback BEFORE IPC. os.replace publishes every result whole.
    output = OUTPUT_DIR / (args.query_id + "_output.json")
    atomic_json(output, {"answer": "", "citations": [], "confidence": 0.0})
    # Linux watchdog is independent of the socket timeout. The complete fallback
    # already exists, so even a stuck client can exit without writing in a signal
    # handler. This is not a claim to control OS scheduling or filesystem stalls.
    watchdog = hasattr(signal, "setitimer")
    if watchdog:
        signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
        signal.setitimer(signal.ITIMER_REAL, 27.0)
    try:
        text, citations, confidence = answer(args.corpus, args.query)
        atomic_json(output, {"answer": text, "citations": citations, "confidence": confidence})
    finally:
        if watchdog:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
