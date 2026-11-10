"""Container healthcheck: exits 0 if this worker's liveness key exists in Redis.

Usage: ``python -m transcriber.worker.healthcheck``
"""

import asyncio
import sys

from redis.asyncio import Redis

from transcriber.core.config import get_settings
from transcriber.jobs.store import JobStore


async def _is_alive() -> bool:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, socket_timeout=3, decode_responses=True)
    try:
        store = JobStore(
            redis,
            result_ttl_seconds=settings.result_ttl_seconds,
            max_queue_size=settings.max_queue_size,
        )
        return await store.is_worker_alive(settings.worker_id)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(_is_alive()) else 1)
