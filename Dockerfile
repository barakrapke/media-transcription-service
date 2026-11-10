# One image for both roles: the API (default command) and the worker (`python -m transcriber.worker`).

# ---- Build: install dependencies into a virtualenv with uv ----------------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first, so code changes don't invalidate this (slow) layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# GPU builds only (docker-compose.gpu.yml): CUDA libraries used by CTranslate2.
ARG WITH_CUDA=false
RUN if [ "$WITH_CUDA" = "true" ]; then uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"; fi

# ---- Runtime: Python + ffmpeg + the virtualenv; no build tools ------------------------------
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data/uploads /models \
    && chown -R app:app /data /models

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    HF_HOME=/models

USER app
WORKDIR /app
EXPOSE 8000
CMD ["python", "-m", "transcriber.api"]
