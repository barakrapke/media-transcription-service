"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

from transcriber import __version__
from transcriber.api.errors import register_exception_handlers
from transcriber.api.middleware import MaxBodySizeMiddleware, RequestIdMiddleware
from transcriber.api.routes import health, transcriptions
from transcriber.core.config import Settings, get_settings
from transcriber.jobs.store import JobStore
from transcriber.media.storage import UploadStorage

STATIC_DIR = Path(__file__).parent / "static"

# Allowance for multipart framing and the small form fields around the file itself.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024

DESCRIPTION = """
Upload an audio or video file and receive a timestamped transcription.

Transcription runs asynchronously: `POST /v1/transcriptions` returns a job immediately; follow it
with `GET /v1/transcriptions/{id}` (polling) or `GET /v1/transcriptions/{id}/events` (SSE).
"""


def create_app(settings: Settings | None = None, *, redis: Redis | None = None) -> FastAPI:
    settings = settings or get_settings()
    owns_redis = redis is None
    redis_client = redis or Redis.from_url(settings.redis_url, decode_responses=True)
    storage = UploadStorage(settings.upload_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        storage.ensure_root()
        yield
        if owns_redis:
            await redis_client.aclose()

    app = FastAPI(
        title="Media Transcription Service",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.storage = storage
    app.state.store = JobStore(
        redis_client,
        result_ttl_seconds=settings.result_ttl_seconds,
        max_queue_size=settings.max_queue_size,
    )

    register_exception_handlers(app)
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
        limit_mb=settings.max_upload_mb,
        paths={transcriptions.router.prefix},
    )
    app.add_middleware(RequestIdMiddleware)  # added last = outermost

    app.include_router(health.router)
    app.include_router(transcriptions.router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
