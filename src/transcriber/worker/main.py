"""Worker process: loads the model once, then claims and processes jobs one at a time.

Background tasks:

* liveness   - refreshes ``worker:{id}`` (readiness probe / container healthcheck)
* heartbeat  - refreshes the heartbeat of the job currently being processed
* sweeper    - re-queues jobs orphaned by dead workers and removes stale upload files
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError

from transcriber.core.config import Settings
from transcriber.core.logging import job_id_var
from transcriber.engine.base import Transcriber
from transcriber.jobs.store import JobStore
from transcriber.media.storage import UploadStorage
from transcriber.worker.processor import JobProcessor

logger = logging.getLogger(__name__)

_RETRY_DELAY_SECONDS = 2


class Worker:
    def __init__(
        self,
        store: JobStore,
        storage: UploadStorage,
        transcriber: Transcriber,
        settings: Settings,
    ):
        self._store = store
        self._storage = storage
        self._settings = settings
        self._processor = JobProcessor(store, storage, transcriber, settings)
        self._stopping = asyncio.Event()

    def request_stop(self) -> None:
        if not self._stopping.is_set():
            logger.info("Shutdown requested; finishing the current job before exiting")
            self._stopping.set()

    async def run(self) -> None:
        background = [
            asyncio.create_task(
                self._every(self._settings.heartbeat_interval_seconds, self._liveness)
            ),
            asyncio.create_task(self._every(self._settings.sweep_interval_seconds, self.sweep)),
        ]
        logger.info("Worker %s ready; waiting for jobs", self._settings.worker_id)
        try:
            while not self._stopping.is_set():
                try:
                    job_id = await self._store.claim(
                        timeout=self._settings.claim_timeout_seconds,
                        heartbeat_ttl=self._settings.heartbeat_ttl_seconds,
                    )
                except RedisError:
                    logger.exception("Could not claim a job; retrying")
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                if job_id is not None:
                    await self._process_with_heartbeat(job_id)
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            with contextlib.suppress(RedisError):
                await self._store.unregister_worker(self._settings.worker_id)
            logger.info("Worker stopped")

    async def _process_with_heartbeat(self, job_id: str) -> None:
        token = job_id_var.set(job_id)
        heartbeat = asyncio.create_task(
            self._every(
                self._settings.heartbeat_interval_seconds,
                lambda: self._store.heartbeat(job_id, self._settings.heartbeat_ttl_seconds),
            )
        )
        try:
            await self._processor.process(job_id)
        except RedisError:
            # e.g. Redis went away before the job started or while recording a failure. The
            # job is recovered by the sweeper; this worker keeps running and retries claiming.
            logger.exception("Redis error while handling the job; it will be recovered")
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            job_id_var.reset(token)

    async def _liveness(self) -> None:
        await self._store.register_worker(
            self._settings.worker_id, self._settings.heartbeat_ttl_seconds
        )

    async def sweep(self) -> None:
        requeued, failed, dropped = await self._store.recover_orphans(
            heartbeat_ttl=self._settings.heartbeat_ttl_seconds,
            max_attempts=self._settings.max_attempts,
        )
        if requeued or failed or dropped:
            logger.warning(
                "Recovered orphaned jobs: requeued=%d failed=%d dropped=%d",
                requeued,
                failed,
                dropped,
            )
        await self._remove_stale_uploads()

    async def _remove_stale_uploads(self) -> None:
        """Delete upload files left behind (e.g. by a crash) whose job is gone or finished."""
        for job_id in self._storage.list_older_than(self._settings.result_ttl_seconds):
            job = await self._store.get(job_id)
            if job is None or job.status.is_terminal:
                logger.info("Removing stale upload %s", job_id)
                self._storage.delete(job_id)

    @staticmethod
    async def _every(interval: float, action: Callable[[], Awaitable[object]]) -> None:
        while True:
            try:
                await action()
            except Exception:
                logger.exception("Periodic task failed")
            await asyncio.sleep(interval)


def load_transcriber(settings: Settings) -> Transcriber:
    from transcriber.engine.faster_whisper import FasterWhisperTranscriber

    transcriber = FasterWhisperTranscriber(settings)
    if settings.model_warmup:
        transcriber.warmup()
    return transcriber


async def run_worker(settings: Settings) -> None:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = JobStore(
        redis,
        result_ttl_seconds=settings.result_ttl_seconds,
        max_queue_size=settings.max_queue_size,
    )
    storage = UploadStorage(settings.upload_dir)
    storage.ensure_root()

    # Model download/load is blocking and can take minutes on first start.
    transcriber = await asyncio.to_thread(load_transcriber, settings)
    worker = Worker(store, storage, transcriber, settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.request_stop)
    try:
        await worker.run()
    finally:
        await redis.aclose()
