"""Container healthcheck: exits 0 if this worker's liveness key exists in Redis.

Usage: ``python -m transcriber.worker.healthcheck``
"""

import asyncio
import sys

from redis.asyncio import Redis

from transcriber.core.config import get_settings
from transcriber.jobs.store import WORKER_PREFIX


async def _is_alive() -> bool:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, socket_timeout=3)
    try:
        return bool(await redis.exists(f"{WORKER_PREFIX}{settings.worker_id}"))
    finally:
        await redis.aclose()


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(_is_alive()) else 1)
