# MC3 implementation

The full implementation is in `starter/app/`; build from `starter/`. It follows
the flag-based invocations and `_output.json` filenames in `CONTRACT.md`.

## Architecture

`server.py` is the container CMD. It loads Qwen2.5-7B-Instruct in bfloat16 and
MiniLM embeddings onto the AMD GPU, performs warm-up, then opens a private Unix
socket. There is no HTTP service or exposed TCP port. Failure to find ROCm or an
AMD GPU fails startup; there is no CPU fallback for production models.

`app.py --index` parses the corpus once, while server startup can proceed in
parallel. Each file runs in an isolated, time-limited process. Local Tesseract
handles images and scanned PDFs. Spreadsheet/CSV rows repeat their headers;
DOCX text/tables, PDFs, source files and logs retain source locations. Encrypted,
unreadable, unknown, oversize, malformed and timed-out files are logged and
skipped. Parsing has explicit page, row, memory, size and total-time limits;
those safeguards can omit content from unusually large documents. Inspect
`/app/index/generation-*/parse_errors.json` before submission.

OCR preserves word-box alignment and removes long frame/grid rules when a large
frame is detected. This matters for the supplied connector diagram: ordinary
Tesseract segmentation omits the boxed pin labels. The preprocessing recovers
those labels while retaining their positions above the signal names.

The server builds persistent SQLite FTS5/BM25 data, a normalized float32 vector
matrix, and exact chunk-to-file mappings. The matrix provides exact cosine
search without requiring a Python 3.14 FAISS wheel. Keyword/dense rankings are
combined, and identifiers in retrieved evidence trigger cross-file searches.
Withdrawn paths are demoted for current questions but retained for verification.

Every query executes a fresh `app.py` process that imports only the standard
library and the small IPC module. It writes a complete empty result first, then
contacts the resident server. A 26-second absolute IPC deadline and a Linux
27-second watchdog bound the client's wait. Atomic replacement prevents partial
JSON. The server has its own 24-second inference allowance and per-generation
stopping criteria. A timeout returns the already-valid empty output. These are
operational bounds, not a mathematical guarantee against OS suspension, blocked
storage, process startup delays, or GPU driver failure.

The model proposes a short value and source quotes. Python checks that the
quotes actually occur in the cited chunks and that the complete value occurs in
the evidence. A separate model pass checks the question, applicability and
minimum necessary evidence set. Only the validated evidence paths become
citations. This does not prove semantic correctness: arbitrary OCR, retrieval
and model mistakes remain possible. No claim of 100% hidden-corpus accuracy is
made. The implementation handles extractive value questions; a new numeric
value requiring arithmetic is deliberately refused.

## Offline dependencies and ROCm

The Dockerfile retains the exact mandated base. Requirements are explicitly
pinned and installed with `--no-deps` into an overlay directory. Torch,
torchvision and the base Pillow installation are excluded. Build checks compare
the base torch/torchvision identity and installed-file hashes before and after
dependency installation. Import smoke checks and dependency validation run at
build time.

Both model snapshots are downloaded at pinned revisions during `docker build`;
Tesseract and its English language data are installed then too. Safetensors-only
downloads avoid duplicate checkpoint formats. All runtime model loads use local
paths and `local_files_only=True`, with offline environment variables enabled.
The final uncompressed image size must still be measured on the build host.

## Build and evaluate on Linux with an AMD GPU

From the repository root:

```bash
./setup.sh
docker build -t mc3-rag:latest starter
docker run -d --name mc3-rag --network none \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-drop DAC_OVERRIDE --shm-size=8g mc3-rag:latest
docker cp mc3-corpus/. mc3-rag:/app/corpus/
docker exec mc3-rag python3 /app/app.py --index /app/corpus
python3 evaluate_sample.py mc3-rag
docker logs mc3-rag
docker inspect mc3-rag:latest --format '{{.Size}}'
```

Monitor GPU memory with `amd-smi` during startup, indexing and all ten questions.
The kit's `selfcheck.py` starts its own container without GPU device flags; on
ordinary Docker installations that container will not see the GPU. Use a GPU
enabled Docker runtime configuration or add the same device flags to its
`docker run` command before relying on its result. The check is left unchanged.

The stock contract requires a 60 GiB uncompressed image, 1–48 GiB GPU memory,
600-second startup and 30 seconds per query. The local Windows host cannot
validate these Linux/ROCm runtime properties.

Local logic tests (no model downloads):

```bash
python -m unittest discover -s tests -v
```

`evaluate_sample.py` belongs outside the submitted image. It compares each
sample answer against the supplied aliases and demands the exact citation set.
It never supplies expected answers to the pipeline.

## Validation completed on this host

- 31 tests run: 30 passed, 1 skipped because this Windows Python lacks AF_UNIX.
- Actual PDF extraction and encryption checks passed using workspace-local
  PyMuPDF 1.26.7; the container's pinned version is checked during its build.
- Portable Tesseract/WASM with local English data read REV-C2 from the asset
  label and B11 through B16 from the preprocessed pin diagram. The TSV layout
  preserves B14 above THERM_ALERT#. This is a real OCR check, but does not
  replace testing native Tesseract inside the final Linux image.
- Python compilation and requirements protection checks passed.
- Docker build, native Linux OCR, ROCm inference, image size, GPU memory,
  full sample answer accuracy and target-GPU latency remain unverified: the
  Docker engine is not running on this host and no ROCm model is loaded here.

Model references: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
