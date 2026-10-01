# Layer identity is checked by the evaluator; retain this exact mandated base.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# The overlay holds only the explicit application dependency closure. No base
# package is uninstalled, and pip never resolves accelerator dependencies.
COPY app/requirements.txt app/build_checks.py /app/
RUN python3 /app/build_checks.py snapshot /app/base-rocm.json \
    && python3 /app/build_checks.py requirements /app/requirements.txt \
    && python3 -m pip install --no-cache-dir --no-deps --only-binary=:all: \
         --target /opt/mc3-deps -r /app/requirements.txt

ENV PYTHONPATH=/opt/mc3-deps \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# OCR runs only at indexing time. English and orientation data ship offline.
RUN sed -i 's/http:/https:/g' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 /app/build_checks.py verify /app/base-rocm.json /app/requirements.txt

# Download only safetensors checkpoint formats; avoid a second cache of weights.
# Model revisions in download_models.py are immutable commits.
COPY app/download_models.py /app/download_models.py
RUN HF_HOME=/tmp/mc3-build-hf python3 /app/download_models.py /models \
    && rm -rf /tmp/mc3-build-hf \
    && python3 /app/build_checks.py models /models

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/mc3-hf \
    MC3_LLM_MODEL=/models/qwen \
    MC3_EMBED_MODEL=/models/embedding \
    MC3_INDEX_DIR=/app/index \
    MC3_OUTPUT_DIR=/app/output \
    MC3_SOCKET=/tmp/mc3.sock

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index \
    && python3 -m compileall -q /app

# One resident model process, a private Unix socket, no network port.
CMD ["python3", "/app/server.py"]
