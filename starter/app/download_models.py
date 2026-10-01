#!/usr/bin/env python3
"""Build-time-only download of immutable, safetensors-only model snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


MODELS = (
    {
        "directory": "qwen",
        "repository": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "files": [
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE",
            "model.safetensors.index.json", "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
    },
    {
        "directory": "embedding",
        "repository": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "files": [
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.txt", "model.safetensors",
            "sentence_bert_config.json", "1_Pooling/config.json", "README.md",
        ],
    },
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download

    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model in MODELS:
        directory = root / model["directory"]
        snapshot_download(
            repo_id=model["repository"], revision=model["revision"],
            local_dir=str(directory), allow_patterns=model["files"],
            max_workers=4,
        )
        files = {}
        for name in model["files"]:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty model asset: {path}")
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest.append({
            "directory": model["directory"], "repository": model["repository"],
            "revision": model["revision"], "files": files,
        })
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(asset["bytes"] for model in manifest for asset in model["files"].values())
    print(f"Shipped checkpoint assets: {total / 2**30:.2f} GiB. Verify final image layer size separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
