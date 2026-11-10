"""Processing of a single claimed job: decode -> transcribe -> publish segments -> complete."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from redis.exceptions import RedisError

from transcriber.core.config import Settings
from transcriber.core.errors import (
    ErrorCode,
    JobFailure,
    JobTimeoutError,
    MediaTooLongError,
    TranscriptionError,
)
from transcriber.engine.base import Transcriber, TranscriptionInfo
from transcriber.jobs.models import Job, Segment, TranscriptionSummary
from transcriber.jobs.store import JobStore
from transcriber.media.decode import decode_to_wav, load_wav, wav_duration
from transcriber.media.storage import UploadStorage

logger = logging.getLogger(__name__)


@dataclass
class _Event:
    kind: Literal["info", "segment", "done", "error"]
    info: TranscriptionInfo | None = None
    segment: Segment | None = None
    error: BaseException | None = None


class JobStopped(Exception):
    """The job left the ``processing`` state (cancelled or recovered) while being transcribed."""


class JobProcessor:
    def __init__(
        self,
        store: JobStore,
        storage: UploadStorage,
        transcriber: Transcriber,
        settings: Settings,
    ):
        self._store = store
        self._storage = storage
        self._transcriber = transcriber
        self._settings = settings
        self._inference: asyncio.Future[None] | None = None

    async def process(self, job_id: str) -> None:
        job = await self._store.get(job_id)
        if job is None:
            logger.warning("Claimed job no longer exists; skipping")
            await self._store.ack(job_id)
            self._storage.delete(job_id)
            return
        attempt = await self._store.start(job_id)
        if attempt is None:
            logger.info("Job is no longer queued (status=%s); skipping", job.status)
            await self._store.ack(job_id)
            return

        logger.info("Processing started (attempt %d)", attempt)
        try:
            await self._run(job, attempt)
        except JobStopped:
            logger.info("Job stopped: it was cancelled or handed to another worker")
        except JobFailure as exc:
            logger.warning("Job failed: %s (%s)", exc.code, exc.message)
            await self._store.fail(job_id, attempt, exc.code, exc.message)
        except RedisError:
            # The outcome can't be recorded. The job stays claimed without a heartbeat once
            # Redis is back, so the sweeper re-queues it; the worker itself stays up.
            logger.exception("Lost the connection to Redis while processing the job")
        except Exception:
            logger.exception("Unexpected error while processing job")
            await self._store.fail(
                job_id, attempt, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred."
            )
        finally:
            await self._release_upload(job_id)
            await self._drain_inference()

    async def _run(self, job: Job, attempt: int) -> None:
        started = time.monotonic()
        deadline = started + self._settings.job_timeout_seconds
        max_seconds = self._settings.max_duration_seconds
        with tempfile.TemporaryDirectory(prefix="transcriber-") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            await decode_to_wav(
                self._storage.path_for(job.id),
                wav_path,
                timeout=min(self._settings.decode_timeout_seconds, deadline - time.monotonic()),
                max_seconds=max_seconds,
            )
            # The API rejects long media up front, but only when ffprobe reports a duration;
            # the decoded audio is the authoritative length.
            if wav_duration(wav_path) > max_seconds:
                raise MediaTooLongError(
                    f"The media is longer than the maximum of {max_seconds / 60:.0f} minutes."
                )
            audio = await asyncio.to_thread(load_wav, wav_path)

        info, text = await self._transcribe(job, attempt, audio, deadline)
        summary = TranscriptionSummary(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            model=self._transcriber.model_name,
            processing_time=round(time.monotonic() - started, 2),
        )
        if await self._store.complete(job.id, attempt, summary):
            logger.info(
                "Processing completed in %.1fs (audio %.1fs, language=%s)",
                summary.processing_time,
                info.duration,
                info.language,
            )

    async def _release_upload(self, job_id: str) -> None:
        """Delete the upload once the job is finished. An active job keeps it: it was handed to
        another worker, or its outcome couldn't be recorded and it will be retried."""
        try:
            job = await self._store.get(job_id)
        except RedisError:
            logger.warning("Keeping the upload: job state is unknown while Redis is unavailable")
            return
        if job is None or job.status.is_terminal:
            self._storage.delete(job_id)

    async def _drain_inference(self) -> None:
        """Wait for the inference thread of the last job to exit before taking another job, so
        two transcriptions never run on the shared model at the same time."""
        inference, self._inference = self._inference, None
        if inference is None or inference.done():
            return
        logger.info("Waiting for the model to finish its current segment")
        await inference

    async def _transcribe(
        self, job: Job, attempt: int, audio: npt.NDArray[np.float32], deadline: float
    ) -> tuple[TranscriptionInfo, str]:
        """Run the (blocking) model in a thread and consume its segments on the event loop.

        The thread pushes events into an asyncio queue; the loop persists each segment, which
        also tells us whether the job is still ``processing``. Cancellation and the job deadline
        are honoured between segments by setting ``stop`` - CTranslate2 itself can't be
        interrupted mid-segment, so a stopped thread is drained later by ``_drain_inference``
        (after the job's outcome has been recorded).
        """
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[_Event] = asyncio.Queue()
        stop = threading.Event()

        def emit(event: _Event) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        def run() -> None:
            try:
                info, segments = self._transcriber.transcribe(
                    audio, language=job.language, word_timestamps=job.word_timestamps
                )
                emit(_Event("info", info=info))
                for segment in segments:
                    if stop.is_set():
                        return
                    emit(_Event("segment", segment=segment))
                emit(_Event("done"))
            except BaseException as exc:
                emit(_Event("error", error=exc))

        thread = loop.run_in_executor(None, run)
        info: TranscriptionInfo | None = None
        texts: list[str] = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise JobTimeoutError()
                try:
                    event = await asyncio.wait_for(events.get(), timeout=remaining)
                except TimeoutError:
                    raise JobTimeoutError() from None

                if event.kind == "error":
                    logger.error("Model failed", exc_info=event.error)
                    raise TranscriptionError()
                if event.kind == "info":
                    info = event.info
                elif event.kind == "segment" and event.segment is not None and info is not None:
                    segment = event.segment
                    progress = min(segment.end / info.duration, 0.99) if info.duration else 0.0
                    if not await self._store.append_segment(job.id, attempt, segment, progress):
                        raise JobStopped()
                    if segment.text:
                        texts.append(segment.text)
                elif event.kind == "done" and info is not None:
                    return info, " ".join(texts)
        finally:
            stop.set()
            self._inference = thread
