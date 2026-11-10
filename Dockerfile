# syntax=docker/dockerfile:1.7
# One image, two roles: the API (default command) and the worker (`python -m transcriber.worker`).

ARG PYTHON_VERSION=3.12

# ---- base: runtime OS dependencies shared by every stage ------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/models \
    DATA_DIR=/data
WORKDIR /app

# ---- builder: resolve dependencies into a virtualenv ----------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable
# Optional CUDA runtime libraries for GPU inference (see docker-compose.gpu.yml).
ARG WITH_CUDA=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_CUDA" = "true" ]; then \
      uv pip install --python /app/.venv/bin/python "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"; \
    fi

# ---- dev: adds test/lint tooling; used by `make test-docker` ---------------------------------
FROM base AS dev
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
COPY tests ./tests
CMD ["pytest"]

# ---- runtime (default target) ---------------------------------------------------------------
FROM base AS runtime
COPY --from=builder /app/.venv /app/.venv
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
RUN mkdir -p /data/uploads /models && chown -R app:app /data /models
USER app
EXPOSE 8000
CMD ["python", "-m", "transcriber.api"]
