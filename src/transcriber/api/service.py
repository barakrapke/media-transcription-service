"""Use cases behind the HTTP routes. Routes stay thin; all decisions live here."""

from __future__ import annotations

import logging
import unicodedata
import uuid
from pathlib import PurePath

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from transcriber.api.schemas import JobResponse
from transcriber.core.config import Settings
from transcriber.core.errors import (
    DurationTooLongError,
    JobNotCompletedError,
    JobNotFoundError,
    NoAudioStreamError,
    QueueFullError,
    UnsupportedLanguageError,
)
from transcriber.engine.languages import SUPPORTED_LANGUAGES
from transcriber.formatters import EXPORT_FORMATS
from transcriber.jobs.models import Job, JobStatus, utc_now
from transcriber.jobs.store import JobStore
from transcriber.media.probe import probe
from transcriber.media.storage import UploadStorage

logger = logging.getLogger(__name__)

_MAX_FILENAME_LENGTH = 255


def normalize_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if value in ("", "auto"):
        return None
    if value not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(f"Language '{value}' is not supported.")
    return value


def clean_filename(filename: str | None) -> str:
    """Keep only the base name, drop control characters. Used for display only."""
    name = PurePath((filename or "").replace("\\", "/")).name
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    return name[:_MAX_FILENAME_LENGTH] or "upload"


def is_valid_job_id(job_id: str) -> bool:
    return len(job_id) == 32 and all(ch in "0123456789abcdef" for ch in job_id)


class TranscriptionService:
    def __init__(self, store: JobStore, storage: UploadStorage, settings: Settings):
        self._store = store
        self._storage = storage
        self._settings = settings

    async def create(
        self, upload: UploadFile, language: str | None, word_timestamps: bool
    ) -> JobResponse:
        language = normalize_language(language)
        if await self._store.is_queue_full():  # cheap early reject; re-checked atomically below
            raise QueueFullError(headers={"Retry-After": "30"})

        job_id = uuid.uuid4().hex
        try:
            size = await run_in_threadpool(
                self._storage.save, job_id, upload.file, self._settings.max_upload_bytes
            )
            info = await probe(self._storage.path_for(job_id), self._settings.probe_timeout_seconds)
            if not info.has_audio:
                raise NoAudioStreamError()
            limit = self._settings.max_duration_seconds
            if info.duration is not None and info.duration > limit:
                raise DurationTooLongError(
                    f"The media is {info.duration / 60:.0f} minutes long; "
                    f"the maximum is {limit / 60:.0f} minutes."
                )
            job = Job(
                id=job_id,
                status=JobStatus.QUEUED,
                filename=clean_filename(upload.filename),
                language=language,
                word_timestamps=word_timestamps,
                created_at=utc_now(),
            )
            await self._store.create(job)
        except BaseException:
            self._storage.delete(job_id)
            raise

        logger.info(
            "Job %s queued (%s, %.1f MB, %s, %ss)",
            job_id,
            info.format_name,
            size / 1_048_576,
            language or "auto",
            info.duration,
        )
        return JobResponse.from_job(job, [])

    async def _get_job(self, job_id: str) -> Job:
        job = await self._store.get(job_id) if is_valid_job_id(job_id) else None
        if job is None:
            raise JobNotFoundError()
        return job

    async def ensure_exists(self, job_id: str) -> None:
        await self._get_job(job_id)

    async def get(self, job_id: str) -> JobResponse:
        job = await self._get_job(job_id)
        segments = (
            await self._store.get_segments(job_id) if job.status is JobStatus.COMPLETED else []
        )
        return JobResponse.from_job(job, segments)

    async def export(self, job_id: str, fmt: str) -> tuple[str, str, str]:
        """Return (content, media type, download filename) for a completed job."""
        job = await self._get_job(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise JobNotCompletedError(
                f"The job is {job.status.value}; exports are available once it has completed."
            )
        render, media_type, extension = EXPORT_FORMATS[fmt]
        segments = await self._store.get_segments(job_id)
        return render(segments), media_type, f"{PurePath(job.filename).stem}.{extension}"

    async def delete(self, job_id: str) -> JobResponse | None:
        """Cancel an active job (returns its new state) or delete a finished one (returns None)."""
        job = await self._get_job(job_id)
        if not job.status.is_terminal and await self._store.cancel(job_id):
            if job.status is JobStatus.QUEUED:
                await self._store.remove_from_pending(job_id)
                self._storage.delete(job_id)
            # A processing job is stopped by its worker, which also removes the upload.
            logger.info("Job %s cancelled", job_id)
            return await self.get(job_id)
        await self._store.delete(job_id)
        self._storage.delete(job_id)
        logger.info("Job %s deleted", job_id)
        return None
