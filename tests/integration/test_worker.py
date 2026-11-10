import asyncio
import shutil
import uuid
from pathlib import Path

import pytest

from tests.conftest import FakeTranscriber, MediaFiles
from transcriber.core.config import Settings
from transcriber.core.errors import ErrorCode
from transcriber.jobs.models import Job, JobStatus, Segment, utc_now
from transcriber.jobs.store import JobStore, job_key
from transcriber.media.storage import UploadStorage
from transcriber.worker.main import Worker
from transcriber.worker.processor import JobProcessor


async def enqueue(store: JobStore, storage: UploadStorage, source: Path, **fields: object) -> str:
    job = Job.model_validate(
        {
            "id": uuid.uuid4().hex,
            "status": JobStatus.QUEUED,
            "filename": source.name,
            "created_at": utc_now(),
            **fields,
        }
    )
    shutil.copy(source, storage.path_for(job.id))
    await store.create(job)
    return job.id


async def claim(store: JobStore) -> str:
    job_id = await store.claim(timeout=1, heartbeat_ttl=5)
    assert job_id is not None
    return job_id


@pytest.fixture
def processor(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber, settings: Settings
) -> JobProcessor:
    return JobProcessor(store, storage, transcriber, settings)


async def test_success(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    job_id = await enqueue(store, storage, media.stereo, language="en", word_timestamps=True)
    await processor.process(await claim(store))

    job = await store.get(job_id)
    assert job is not None and job.summary is not None
    assert job.status is JobStatus.COMPLETED
    assert job.summary.text == "Hello world. This is a test."
    assert job.summary.model == "fake-model"
    assert len(await store.get_segments(job_id)) == 2
    assert transcriber.calls == [
        {"samples": pytest.approx(16000, abs=200), "language": "en", "word_timestamps": True}
    ]
    assert not storage.path_for(job_id).exists()  # upload cleaned up


async def test_silence_completes_with_empty_transcript(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    transcriber.segments = []
    job_id = await enqueue(store, storage, media.tone)
    await processor.process(await claim(store))
    job = await store.get(job_id)
    assert job is not None and job.summary is not None
    assert (job.status, job.summary.text) == (JobStatus.COMPLETED, "")


async def test_decode_error_fails_job(
    processor: JobProcessor, store: JobStore, storage: UploadStorage, media: MediaFiles
) -> None:
    job_id = await enqueue(store, storage, media.corrupt)
    await processor.process(await claim(store))
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert (job.status, job.error.code) == (JobStatus.FAILED, ErrorCode.DECODE_ERROR)
    assert not storage.path_for(job_id).exists()


async def test_model_error_fails_job(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    transcriber.error = RuntimeError("CUDA out of memory")
    job_id = await enqueue(store, storage, media.tone)
    await processor.process(await claim(store))
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert job.error.code == ErrorCode.TRANSCRIPTION_ERROR
    assert "CUDA" not in job.error.message  # internals are not leaked to clients


async def test_job_timeout_waits_for_the_model_before_returning(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber,
    settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    transcriber.delay_per_segment = 1.5
    processor = JobProcessor(
        store, storage, transcriber, settings.model_copy(update={"job_timeout_seconds": 1})
    )
    job_id = await enqueue(store, storage, media.tone)
    await processor.process(await claim(store))
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert job.error.code == ErrorCode.JOB_TIMEOUT
    # The next job must never run on the shared model while this one is still inferring.
    assert transcriber.finished.is_set()


async def test_media_longer_than_limit_is_rejected_after_decoding(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber,
    settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    """ffprobe can't always tell the duration (e.g. browser-recorded WebM); decoding can."""
    processor = JobProcessor(
        store, storage, transcriber, settings.model_copy(update={"max_duration_seconds": 1})
    )
    job_id = await enqueue(store, storage, media.tone)  # 2 s
    await processor.process(await claim(store))
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert job.error.code == ErrorCode.DURATION_TOO_LONG
    assert transcriber.calls == []


async def test_cancel_while_processing(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    transcriber.segments = [Segment(id=i, start=i, end=i + 1, text=f"s{i}") for i in range(50)]
    transcriber.delay_per_segment = 0.02
    job_id = await enqueue(store, storage, media.tone)
    task = asyncio.create_task(processor.process(await claim(store)))
    while not await store.get_segments(job_id):
        await asyncio.sleep(0.01)
    assert await store.cancel(job_id)
    await asyncio.wait_for(task, timeout=5)

    job = await store.get(job_id)
    assert job is not None and job.status is JobStatus.CANCELLED
    assert len(await store.get_segments(job_id)) < 50
    assert not storage.path_for(job_id).exists()


async def test_skips_cancelled_or_deleted_jobs(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    cancelled = await enqueue(store, storage, media.tone)
    deleted = await enqueue(store, storage, media.tone)
    await store.cancel(cancelled)
    await store.delete(deleted)
    await processor.process(await claim(store))
    await processor.process(await claim(store))
    assert transcriber.calls == []
    assert await store.redis.llen("queue:processing") == 0


async def test_worker_presumed_dead_stops_writing_after_takeover(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, media: MediaFiles,
) -> None:  # fmt: skip
    """Worker A stalls, the job is re-queued and started by worker B. A must stop without
    writing into B's run and without deleting the upload B needs."""
    transcriber.segments = [Segment(id=i, start=i, end=i + 1, text="A") for i in range(50)]
    transcriber.delay_per_segment = 0.02
    job_id = await enqueue(store, storage, media.tone)
    task = asyncio.create_task(processor.process(await claim(store)))
    while not await store.get_segments(job_id):
        await asyncio.sleep(0.01)
    # The sweeper re-queues the job and worker B immediately starts attempt 2.
    await store.redis.hset(job_key(job_id), "status", "queued")
    await store.redis.delete(f"job:{job_id}:segments")
    assert await store.start(job_id) == 2
    await asyncio.wait_for(task, timeout=5)

    job = await store.get(job_id)
    assert job is not None
    assert (job.status, job.attempts, job.summary) == (JobStatus.PROCESSING, 2, None)
    assert await store.get_segments(job_id) == []  # nothing from attempt 1 leaked in
    assert storage.path_for(job_id).exists()


async def test_redis_failure_mid_job_keeps_worker_and_upload(
    processor: JobProcessor, store: JobStore, storage: UploadStorage,
    media: MediaFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    from redis.exceptions import ConnectionError as RedisConnectionError

    async def broken(*args: object, **kwargs: object) -> bool:
        raise RedisConnectionError("connection reset")

    job_id = await enqueue(store, storage, media.tone)
    monkeypatch.setattr(store, "append_segment", broken)
    await processor.process(await claim(store))  # must not raise

    job = await store.get(job_id)
    assert job is not None and job.status is JobStatus.PROCESSING  # left for the sweeper
    assert storage.path_for(job_id).exists()  # needed by the retry


async def test_worker_survives_redis_errors(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber,
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    from redis.exceptions import ConnectionError as RedisConnectionError

    worker = Worker(store, storage, transcriber, settings)

    async def broken(self: JobProcessor, job_id: str) -> None:
        raise RedisConnectionError("connection reset")

    monkeypatch.setattr(JobProcessor, "process", broken)
    monkeypatch.setattr("transcriber.worker.main._RETRY_DELAY_SECONDS", 0)
    await worker._process_with_heartbeat("0" * 32)  # must not raise


async def test_worker_loop_processes_and_stops(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber,
    settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    worker = Worker(store, storage, transcriber, settings)
    job_id = await enqueue(store, storage, media.tone)
    run = asyncio.create_task(worker.run())
    for _ in range(300):
        job = await store.get(job_id)
        if job is not None and job.status is JobStatus.COMPLETED:
            break
        await asyncio.sleep(0.02)
    assert await store.is_worker_alive(settings.worker_id)
    worker.request_stop()
    await asyncio.wait_for(run, timeout=5)

    job = await store.get(job_id)
    assert job is not None and job.status is JobStatus.COMPLETED
    assert not await store.is_worker_alive(settings.worker_id)


async def test_sweep_recovers_jobs_and_removes_stale_uploads(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber,
    settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    worker = Worker(
        store, storage, transcriber, settings.model_copy(update={"result_ttl_seconds": 0})
    )
    orphan = await enqueue(store, storage, media.tone)
    await claim(store)
    await store.start(orphan)
    await store.redis.delete(f"job:{orphan}:heartbeat")
    await store.redis.hset(job_key(orphan), "claimed_at", "0")
    shutil.copy(media.tone, storage.path_for("deadbeef"))  # upload with no job
    (settings.upload_dir / ".DS_Store").write_bytes(b"")  # stray file must not break the sweep

    await worker.sweep()

    job = await store.get(orphan)
    assert job is not None and job.status is JobStatus.QUEUED
    assert storage.path_for(orphan).exists()  # still needed by the next attempt
    assert not storage.path_for("deadbeef").exists()
