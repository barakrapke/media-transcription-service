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

from transcriber.core.config import Settings
from transcriber.core.errors import ErrorCode, JobFailure, JobTimeoutError, TranscriptionError
from transcriber.engine.base import Transcriber, TranscriptionInfo
from transcriber.jobs.models import Job, Segment, TranscriptionSummary
from transcriber.jobs.store import JobStore
from transcriber.media.decode import decode_to_wav, load_wav
from transcriber.media.storage import UploadStorage

logger = logging.getLogger(__name__)

# How long to wait for the inference thread to notice a stop request before moving on.
_THREAD_STOP_GRACE_SECONDS = 60


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

    async def process(self, job_id: str) -> None:
        job = await self._store.get(job_id)
        if job is None:
            logger.warning("Claimed job no longer exists; skipping")
            await self._store.ack(job_id)
            self._storage.delete(job_id)
            return
        if not await self._store.start(job_id):
            logger.info("Job is no longer queued (status=%s); skipping", job.status)
            await self._store.ack(job_id)
            return

        started = time.monotonic()
        deadline = started + self._settings.job_timeout_seconds
        logger.info("Processing started (attempt %d)", job.attempts + 1)
        keep_upload = False
        try:
            with tempfile.TemporaryDirectory(prefix="transcriber-") as tmp:
                wav_path = Path(tmp) / "audio.wav"
                await decode_to_wav(
                    self._storage.path_for(job_id),
                    wav_path,
                    timeout=min(self._settings.decode_timeout_seconds, deadline - time.monotonic()),
                )
                audio = await asyncio.to_thread(load_wav, wav_path)
            info, text = await self._transcribe(job, audio, deadline)
            summary = TranscriptionSummary(
                text=text,
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                model=self._transcriber.model_name,
                processing_time=round(time.monotonic() - started, 2),
            )
            if await self._store.complete(job_id, summary):
                logger.info(
                    "Processing completed in %.1fs (audio %.1fs, language=%s)",
                    summary.processing_time,
                    info.duration,
                    info.language,
                )
        except JobStopped:
            current = await self._store.get(job_id)
            # If the job was re-queued (this worker was presumed dead), the next worker needs
            # the upload; only a cancelled or deleted job releases it.
            keep_upload = current is not None and not current.status.is_terminal
            logger.info("Job stopped while processing (status=%s)", current and current.status)
        except JobFailure as exc:
            logger.warning("Job failed: %s (%s)", exc.code, exc.message)
            await self._store.fail(job_id, exc.code, exc.message)
        except Exception:
            logger.exception("Unexpected error while processing job")
            await self._store.fail(
                job_id, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred during processing."
            )
        finally:
            if not keep_upload:
                self._storage.delete(job_id)

    async def _transcribe(
        self, job: Job, audio: npt.NDArray[np.float32], deadline: float
    ) -> tuple[TranscriptionInfo, str]:
        """Run the (blocking) model in a thread and consume its segments on the event loop.

        The thread pushes events into an asyncio queue; the loop persists each segment, which
        also tells us whether the job is still ``processing``. Cancellation and the job deadline
        are honoured between segments by setting ``stop`` - CTranslate2 itself can't be
        interrupted mid-segment.
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
                    if not await self._store.append_segment(job.id, segment, progress):
                        raise JobStopped()
                    if segment.text:
                        texts.append(segment.text)
                elif event.kind == "done" and info is not None:
                    return info, " ".join(texts)
        finally:
            stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(thread), _THREAD_STOP_GRACE_SECONDS)
            except TimeoutError:
                logger.warning("Inference thread did not stop within the grace period")
