"""Persisted FTS5 BM25 + exact dense-vector retrieval, with identifier hops."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

from protocol import atomic_json

STOP = set("a an the what which who where when how is are was were be been of for to in on at by with and or does do did it its this that from as has have had please tell me".split())
IDENTIFIER = re.compile(r"\b(?:[A-Za-z][A-Za-z0-9]{1,15}[-_]\d[A-Za-z0-9._-]*|[A-Z]{1,8}\d{2,}[A-Za-z0-9-]*)\b")


def identifiers(text: str) -> list[str]:
    return list(dict.fromkeys(value.rstrip("._-").casefold() for value in IDENTIFIER.findall(text)))


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    # TQ-40, tq40 and TQ_40 share a token; identifiers are also split into words.
    compact = re.findall(r"\b[a-z]+[-_]\d[\w-]*", text)
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return [w for w in words if w not in STOP] + [re.sub(r"\W|_", "", w) for w in compact]


def chunk_units(units: list[dict], limit: int = 1250) -> list[dict]:
    chunks, seen = [], set()
    for unit in units:
        text = str(unit["text"]).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + limit)
            if end < len(text):
                boundary = text.rfind("\n", start + limit // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            key = (unit["path"], piece)
            if piece and key not in seen:
                seen.add(key)
                chunks.append({"id": len(chunks), "path": unit["path"],
                               "location": f"{unit['location']}; chars {start}-{end}", "text": piece})
            if end >= len(text):
                break
            # Keep enough shared context for adjacent log lines and split prose.
            next_start = max(start + 1, end - 240)
            newline = text.find("\n", next_start, end)
            start = newline + 1 if newline >= 0 else next_start
    return chunks


class DenseEncoder:
    """Small resident GPU encoder alongside the substantive ROCm answer model."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("Dense retrieval requires the mandated ROCm GPU")
        self.device = torch.device("cuda:0")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: list[str], *, deadline: float | None = None):
        import numpy as np
        output = []
        for start in range(0, len(texts), 32):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("embedding deadline exceeded")
            encoded = self.tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                     max_length=256, return_tensors="pt")
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            output.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(output) if output else np.empty((0, self.dimension), dtype="float32")


def build_index(directory: Path, corpus: Path, units: list[dict], encoder,
                *, generation: str, deadline: float | None = None) -> dict:
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    chunks = chunk_units(units)
    database = directory / "chunks.sqlite3"
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, path TEXT, location TEXT, text TEXT)")
        db.execute("CREATE VIRTUAL TABLE search USING fts5(body, path, tokenize='unicode61')")
        db.execute("CREATE TABLE identifiers(value TEXT, chunk_id INTEGER, PRIMARY KEY(value,chunk_id))")
        for chunk in chunks:
            db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (chunk["id"], chunk["path"], chunk["location"], chunk["text"]))
            db.execute("INSERT INTO search(rowid,body,path) VALUES(?,?,?)", (chunk["id"],
                       " ".join(tokens(chunk["text"])), " ".join(tokens(chunk["path"]))))
            for identifier in identifiers(chunk["text"]):
                db.execute("INSERT OR IGNORE INTO identifiers VALUES(?,?)", (identifier, chunk["id"]))
    # A normalized float32 matrix is an exact cosine vector index, without a
    # native FAISS wheel dependency on the mandated Python 3.14 runtime.
    matrix = encoder.encode([f"{c['path']}\n{c['text']}" for c in chunks], deadline=deadline)
    np.save(directory / "vectors.npy", matrix, allow_pickle=False)
    manifest = {"version": 1, "generation": generation, "corpus": str(corpus.resolve()),
                "chunks": len(chunks), "dimensions": int(matrix.shape[1])}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


class Retriever:
    def __init__(self, directory: Path, encoder):
        import numpy as np
        self.np = np
        self.encoder = encoder
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.db = sqlite3.connect((directory / "chunks.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.chunks = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM chunks ORDER BY id")}
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), self.manifest["dimensions"]):
            raise ValueError("corrupt vector/chunk mapping")

    def close(self):
        self.db.close()
        mapping = getattr(self.vectors, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def _lexical(self, query: str, limit: int = 50) -> list[int]:
        terms = list(dict.fromkeys(tokens(query)))[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        return [r[0] for r in self.db.execute(
            "SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search,1.0,0.6) LIMIT ?",
            (expression, limit))]

    def retrieve(self, question: str, *, limit: int = 12, deadline: float | None = None) -> list[dict]:
        if not self.chunks:
            return []
        lexical = self._lexical(question)
        vector = self.encoder.encode([question], deadline=deadline)[0]
        similarities = self.vectors @ vector
        count = min(50, len(self.chunks))
        indices = self.np.argpartition(-similarities, count - 1)[:count]
        dense = sorted((int(i) for i in indices), key=lambda i: (-float(similarities[i]), i))
        scores = defaultdict(float)
        for ranking, weight in ((lexical, 1.2), (dense, 1.0)):
            for rank, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] += weight / (30 + rank)
        historical = bool(re.search(r"\b(old|previous|withdrawn|superseded|historical|original|r\d+|revision\s+\w+)\b", question, re.I))
        for chunk_id in scores:
            if not historical and re.search(r"withdrawn|superseded|obsolete", self.chunks[chunk_id]["path"], re.I):
                scores[chunk_id] *= 0.65
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        # Reserve room for identifier links, including facts whose wording bears
        # no resemblance to the question (incident -> ticket -> fixed version).
        direct, per_file = [], Counter()
        for chunk_id in ranked:
            path = self.chunks[chunk_id]["path"]
            if per_file[path] < 3:
                direct.append(chunk_id)
                per_file[path] += 1
            if len(direct) >= max(1, limit - 4):
                break
        visited = set(direct)
        hops = []
        frontier = direct[:6]
        for _ in range(2):
            found = {}
            for parent in frontier:
                for identifier in identifiers(self.chunks[parent]["text"])[:20]:
                    rows = self.db.execute("SELECT chunk_id FROM identifiers WHERE value=? LIMIT 31", (identifier,)).fetchall()
                    # Ubiquitous product IDs do not establish a useful join.
                    if len(rows) > 30:
                        continue
                    for row in rows:
                        target = row[0]
                        if target not in visited and self.chunks[target]["path"] != self.chunks[parent]["path"]:
                            found[target] = max(found.get(target, 0), scores.get(parent, 0) / max(1, len(rows)))
            frontier = sorted(found, key=lambda i: (-found[i], -scores.get(i, 0), i))[:4]
            for target in frontier:
                visited.add(target)
                hops.append(target)
            if not frontier:
                break
        # Keep both ends of joins early enough to survive the LLM context cap.
        selected = direct[:4] + hops[:4] + direct[4:]
        selected += [i for i in ranked if i not in set(selected)]
        return [dict(self.chunks[i], score=float(scores.get(i, 0))) for i in selected[:limit]]
