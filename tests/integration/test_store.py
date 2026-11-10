import time
import uuid

import pytest

from transcriber.core.errors import ErrorCode, QueueFullError
from transcriber.jobs.models import Job, JobStatus, Segment, TranscriptionSummary, utc_now
from transcriber.jobs.store import PENDING_QUEUE, PROCESSING_QUEUE, JobStore, job_key

SUMMARY = TranscriptionSummary(
    text="hi", language="en", language_probability=0.9, duration=1.0, model="m", processing_time=0.1
)


def new_job(**overrides: object) -> Job:
    fields: dict[str, object] = {
        "id": uuid.uuid4().hex,
        "status": JobStatus.QUEUED,
        "filename": "a.mp3",
        "created_at": utc_now(),
    }
    fields.update(overrides)
    return Job.model_validate(fields)


async def claimed(store: JobStore) -> str:
    job = new_job()
    await store.create(job)
    job_id = await store.claim(timeout=1, heartbeat_ttl=5)
    assert job_id == job.id
    return job_id


async def test_create_get_roundtrip(store: JobStore) -> None:
    job = new_job(language="he", word_timestamps=True)
    await store.create(job)
    loaded = await store.get(job.id)
    assert loaded is not None
    assert loaded.model_dump(exclude={"created_at"}) == job.model_dump(exclude={"created_at"})
    assert loaded.created_at == job.created_at
    assert await store.queue_length() == 1
    assert await store.get("missing") is None


async def test_queue_is_fifo(store: JobStore) -> None:
    first, second = new_job(), new_job()
    await store.create(first)
    await store.create(second)
    assert await store.claim(timeout=1, heartbeat_ttl=5) == first.id
    assert await store.claim(timeout=1, heartbeat_ttl=5) == second.id
    assert await store.claim(timeout=1, heartbeat_ttl=5) is None


async def test_queue_full(store: JobStore) -> None:
    for _ in range(10):
        await store.create(new_job())
    assert await store.is_queue_full()
    job = new_job()
    with pytest.raises(QueueFullError) as exc:
        await store.create(job)
    assert exc.value.headers == {"Retry-After": "30"}
    assert await store.get(job.id) is None  # nothing written when rejected


async def test_lifecycle_to_completed(store: JobStore) -> None:
    job_id = await claimed(store)
    assert await store.start(job_id) == 1
    segment = Segment(id=0, start=0, end=1, text="hi")
    assert await store.append_segment(job_id, 1, segment, 0.5)
    assert await store.complete(job_id, 1, SUMMARY)

    job = await store.get(job_id)
    assert job is not None
    assert job.status is JobStatus.COMPLETED
    assert job.attempts == 1
    assert job.progress == 1
    assert job.summary == SUMMARY
    assert job.started_at is not None and job.completed_at is not None
    assert await store.get_segments(job_id) == [segment]
    assert await store.redis.llen(PROCESSING_QUEUE) == 0  # acked
    assert 0 < await store.redis.ttl(job_key(job_id)) <= 86_400


async def test_terminal_states_are_immutable(store: JobStore) -> None:
    job_id = await claimed(store)
    attempt = await store.start(job_id)
    assert attempt is not None
    assert await store.cancel(job_id)
    assert not await store.complete(job_id, attempt, SUMMARY)
    assert not await store.fail(job_id, attempt, "X", "y")
    segment = Segment(id=0, start=0, end=1, text="x")
    assert not await store.append_segment(job_id, attempt, segment, 0.1)
    assert await store.start(job_id) is None
    job = await store.get(job_id)
    assert job is not None and job.status is JobStatus.CANCELLED


async def test_start_requires_queued(store: JobStore) -> None:
    job_id = await claimed(store)
    assert await store.start(job_id) == 1
    assert await store.start(job_id) is None  # already processing
    assert await store.start("missing") is None
    assert await store.get("missing") is None  # no stray hash created


async def test_fail_records_error(store: JobStore) -> None:
    job_id = await claimed(store)
    assert await store.start(job_id) == 1
    assert await store.fail(job_id, 1, ErrorCode.DECODE_ERROR, "bad")
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert (job.status, job.error.code) == (JobStatus.FAILED, "DECODE_ERROR")


async def test_delete(store: JobStore) -> None:
    job_id = await claimed(store)
    await store.delete(job_id)
    assert await store.get(job_id) is None


async def test_stale_attempt_is_fenced_off(store: JobStore) -> None:
    """A worker presumed dead must not write into the job after another worker took it over."""
    job_id = await claimed(store)
    assert await store.start(job_id) == 1  # worker A
    await store.redis.hset(job_key(job_id), "status", "queued")  # re-queued by the sweeper
    assert await store.start(job_id) == 2  # worker B

    segment = Segment(id=0, start=0, end=1, text="from A")
    assert not await store.append_segment(job_id, 1, segment, 0.5)
    assert not await store.complete(job_id, 1, SUMMARY)
    assert not await store.fail(job_id, 1, "X", "late")
    assert await store.append_segment(job_id, 2, Segment(id=0, start=0, end=1, text="B"), 0.5)
    assert await store.complete(job_id, 2, SUMMARY)
    assert [s.text for s in await store.get_segments(job_id)] == ["B"]


async def test_recover_requeues_job_of_dead_worker(store: JobStore) -> None:
    job_id = await claimed(store)
    await store.start(job_id)
    await store.append_segment(job_id, 1, Segment(id=0, start=0, end=1, text="x"), 0.3)
    await store.redis.delete(f"job:{job_id}:heartbeat")  # the worker died
    await store.redis.hset(job_key(job_id), "claimed_at", str(time.time() - 60))

    assert await store.recover_orphans(heartbeat_ttl=30, max_attempts=2) == (1, 0, 0)
    job = await store.get(job_id)
    assert job is not None
    assert (job.status, job.progress, job.claimed_at) == (JobStatus.QUEUED, 0, None)
    assert await store.get_segments(job_id) == []
    assert await store.redis.lrange(PENDING_QUEUE, 0, -1) == [job_id]
    assert await store.redis.llen(PROCESSING_QUEUE) == 0


async def test_recover_fails_job_after_max_attempts(store: JobStore) -> None:
    job_id = await claimed(store)
    await store.start(job_id)  # attempts = 1
    await store.redis.delete(f"job:{job_id}:heartbeat")
    await store.redis.hset(job_key(job_id), "claimed_at", str(time.time() - 60))

    assert await store.recover_orphans(heartbeat_ttl=30, max_attempts=1) == (0, 1, 0)
    job = await store.get(job_id)
    assert job is not None and job.error is not None
    assert (job.status, job.error.code) == (JobStatus.FAILED, ErrorCode.WORKER_LOST)


async def test_recover_leaves_healthy_and_fresh_claims_alone(store: JobStore) -> None:
    healthy = await claimed(store)  # heartbeat present
    fresh = await claimed(store)
    await store.redis.delete(f"job:{fresh}:heartbeat")
    await store.redis.hdel(job_key(fresh), "claimed_at")  # claimed a moment ago

    assert await store.recover_orphans(heartbeat_ttl=30, max_attempts=2) == (0, 0, 0)
    fresh_job = await store.get(fresh)
    assert fresh_job is not None and fresh_job.claimed_at is not None  # clock started
    healthy_job = await store.get(healthy)
    assert healthy_job is not None and healthy_job.status is JobStatus.QUEUED
    assert await store.redis.llen(PROCESSING_QUEUE) == 2


async def test_recover_drops_stale_entries(store: JobStore) -> None:
    await store.redis.lpush(PROCESSING_QUEUE, "ghost")
    assert await store.recover_orphans(heartbeat_ttl=30, max_attempts=2) == (0, 0, 1)
    assert await store.redis.llen(PROCESSING_QUEUE) == 0


async def test_workers_registry(store: JobStore) -> None:
    assert await store.count_live_workers() == 0
    await store.register_worker("w1", ttl=5)
    await store.register_worker("w2", ttl=5)
    assert await store.count_live_workers() == 2
    assert await store.is_worker_alive("w1")
    await store.unregister_worker("w1")
    assert not await store.is_worker_alive("w1")
