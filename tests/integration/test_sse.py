import asyncio
import json
from pathlib import Path

import httpx

from tests.conftest import FakeTranscriber, MediaFiles
from transcriber.core.config import Settings
from transcriber.jobs.models import Segment
from transcriber.jobs.store import JobStore
from transcriber.media.storage import UploadStorage
from transcriber.worker.processor import JobProcessor

API = "/v1/transcriptions"


def parse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.split("\n\n"):
        lines = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line and line[0] != ":"
        )
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


async def create(client: httpx.AsyncClient, path: Path) -> str:
    with path.open("rb") as fh:
        response = await client.post(API, files={"file": (path.name, fh)})
    return str(response.json()["id"])


async def test_stream_follows_a_job_to_completion(
    client: httpx.AsyncClient, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    transcriber.segments = [Segment(id=i, start=i, end=i + 1, text=f"s{i}") for i in range(5)]
    transcriber.duration = 5
    transcriber.delay_per_segment = 0.03
    job_id = await create(client, media.tone)

    stream = asyncio.create_task(client.get(f"{API}/{job_id}/events"))
    await asyncio.sleep(0.05)  # subscriber connects while the job is still queued
    claimed = await store.claim(timeout=1, heartbeat_ttl=5)
    assert claimed is not None
    await JobProcessor(store, storage, transcriber, settings).process(claimed)
    response = await asyncio.wait_for(stream, timeout=5)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "status" and events[0][1] == {"status": "queued", "attempt": 0}
    assert ("status", {"status": "processing", "attempt": 1}) in events
    assert [data["text"] for name, data in events if name == "segment"] == [
        f"s{i}" for i in range(5)
    ]
    progress = [data["progress"] for name, data in events if name == "progress"]
    assert progress == sorted(progress) and progress[-1] == 1
    assert names[-2:] == ["status", "completed"]
    completed = events[-1][1]
    assert completed["status"] == "completed"
    assert completed["result"]["text"] == "s0 s1 s2 s3 s4"  # type: ignore[index]


async def test_late_subscriber_gets_full_replay(
    client: httpx.AsyncClient, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, settings: Settings, media: MediaFiles,
) -> None:  # fmt: skip
    job_id = await create(client, media.tone)
    claimed = await store.claim(timeout=1, heartbeat_ttl=5)
    assert claimed is not None
    await JobProcessor(store, storage, transcriber, settings).process(claimed)

    events = parse_events((await client.get(f"{API}/{job_id}/events")).text)
    assert [name for name, _ in events] == ["segment", "segment", "progress", "status", "completed"]


async def test_stream_reports_failures(
    client: httpx.AsyncClient, store: JobStore, media: MediaFiles
) -> None:
    job_id = await create(client, media.tone)
    await store.claim(timeout=1, heartbeat_ttl=5)
    attempt = await store.start(job_id)
    assert attempt is not None
    await store.fail(job_id, attempt, "DECODE_ERROR", "bad media")

    events = parse_events((await client.get(f"{API}/{job_id}/events")).text)
    assert events[-1] == ("failed", {"error": {"code": "DECODE_ERROR", "message": "bad media"}})


async def test_stream_ends_when_job_is_deleted(
    client: httpx.AsyncClient, store: JobStore, media: MediaFiles
) -> None:
    job_id = await create(client, media.tone)
    stream = asyncio.create_task(client.get(f"{API}/{job_id}/events"))
    await asyncio.sleep(0.05)
    await store.delete(job_id)
    events = parse_events((await asyncio.wait_for(stream, timeout=5)).text)
    assert events[-1][0] == "failed"
    assert events[-1][1]["error"]["code"] == "JOB_NOT_FOUND"  # type: ignore[index]


async def test_stream_resets_when_job_is_rerun_without_visible_queued_state(
    client: httpx.AsyncClient, store: JobStore, media: MediaFiles
) -> None:
    """Worker A dies mid-job; the sweeper re-queues it and an idle worker B re-claims it before
    the stream polls again. The client must be told to reset and must get all of B's segments."""
    job_id = await create(client, media.tone)
    await store.claim(timeout=1, heartbeat_ttl=5)
    assert await store.start(job_id) == 1
    for i in range(3):
        await store.append_segment(job_id, 1, Segment(id=i, start=i, end=i + 1, text=f"A{i}"), 0.1)

    stream = asyncio.create_task(client.get(f"{API}/{job_id}/events"))
    await asyncio.sleep(0.05)  # the client has received A's three segments
    # Re-queue and immediate re-start, faster than the stream's poll interval.
    await store.redis.hset(f"job:{job_id}", "status", "queued")
    await store.redis.delete(f"job:{job_id}:segments")
    assert await store.start(job_id) == 2
    for i in range(2):
        await store.append_segment(job_id, 2, Segment(id=i, start=i, end=i + 1, text=f"B{i}"), 0.5)
    await store.fail(job_id, 2, "DECODE_ERROR", "end of test")

    events = parse_events((await asyncio.wait_for(stream, timeout=5)).text)
    attempts = [data["attempt"] for name, data in events if name == "status"]
    assert attempts[0] == 1 and 2 in attempts  # the client sees the new attempt
    reset_at = next(i for i, (n, d) in enumerate(events) if n == "status" and d["attempt"] == 2)
    after_reset = [d["text"] for n, d in events[reset_at:] if n == "segment"]
    assert after_reset == ["B0", "B1"]
