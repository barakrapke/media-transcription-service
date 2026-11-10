"""Server-Sent Events stream of a job's progress.

The stream polls the job state in Redis and emits only what changed, so no pub/sub is needed.
On connect it replays the current state and all segments produced so far, which makes late
subscribers and reconnects lossless.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from transcriber.api.schemas import JobResponse
from transcriber.core.errors import ErrorCode
from transcriber.jobs.models import JobStatus
from transcriber.jobs.store import JobStore


def format_event(event: str, data: Any, event_id: str | None = None) -> str:
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


async def job_events(
    store: JobStore,
    job_id: str,
    *,
    poll_interval: float,
    keepalive_interval: float,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    yield "retry: 3000\n\n"
    sent_segments = 0
    last_status: JobStatus | None = None
    last_progress: float | None = None
    last_write = time.monotonic()

    while not await is_disconnected():
        chunks: list[str] = []
        job = await store.get(job_id)
        if job is None:
            yield format_event(
                "failed",
                {
                    "error": {
                        "code": ErrorCode.JOB_NOT_FOUND,
                        "message": "The job no longer exists.",
                    }
                },
            )
            return

        if job.status != last_status:
            if job.status is JobStatus.QUEUED and last_status is not None:
                sent_segments = 0  # re-queued after a worker was lost; segments start over
            chunks.append(format_event("status", {"status": job.status}))
            last_status = job.status

        for segment in await store.get_segments(job_id, start=sent_segments):
            chunks.append(format_event("segment", segment.model_dump(), event_id=str(segment.id)))
            sent_segments += 1

        if job.progress != last_progress:
            chunks.append(format_event("progress", {"progress": job.progress}))
            last_progress = job.progress

        if job.status is JobStatus.COMPLETED:
            response = JobResponse.from_job(job, await store.get_segments(job_id))
            chunks.append(format_event("completed", response.model_dump(mode="json")))
        elif job.status is JobStatus.FAILED and job.error is not None:
            chunks.append(format_event("failed", {"error": job.error.model_dump()}))

        if chunks:
            yield "".join(chunks)
            last_write = time.monotonic()
        elif time.monotonic() - last_write >= keepalive_interval:
            yield ": keepalive\n\n"
            last_write = time.monotonic()

        if job.status.is_terminal:
            return
        await asyncio.sleep(poll_interval)
