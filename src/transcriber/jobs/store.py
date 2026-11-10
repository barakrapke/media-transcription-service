"""Redis-backed job store and reliable work queue.

Key layout (see docs/ARCHITECTURE.md):

* ``job:{id}``            hash  - job fields; JSON for nested values
* ``job:{id}:segments``   list  - segments as produced (live view + SSE replay)
* ``job:{id}:heartbeat``  str   - refreshed by the owning worker, expires on worker death
* ``queue:pending``       list  - job ids waiting for a worker (LPUSH in, pop from the right)
* ``queue:processing``    list  - job ids claimed by a worker (reliable-queue pattern)
* ``worker:{worker_id}``  str   - worker liveness

All state transitions run as Lua scripts so that "check current status, then write" is atomic.
Terminal states are immutable: a write from a stale worker after cancellation or recovery is
rejected instead of silently overwriting the job.

Worker writes are additionally *fenced* by the attempt number returned from ``start``: if a worker
was presumed dead and its job was handed to another worker, the first worker's late writes carry
an outdated attempt and are rejected, so results of two runs can never be mixed.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Mapping
from datetime import datetime
from typing import Any, cast

from redis.asyncio import Redis

from transcriber.core.errors import ErrorCode, QueueFullError
from transcriber.jobs.models import (
    Job,
    JobErrorInfo,
    JobStatus,
    Segment,
    TranscriptionSummary,
    utc_now,
)

PENDING_QUEUE = "queue:pending"
PROCESSING_QUEUE = "queue:processing"
WORKER_PREFIX = "worker:"


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def segments_key(job_id: str) -> str:
    return f"job:{job_id}:segments"


def heartbeat_key(job_id: str) -> str:
    return f"job:{job_id}:heartbeat"


# KEYS: job, pending.  ARGV: max_queue_size, job_id, then field/value pairs.
_CREATE_AND_ENQUEUE = """
if redis.call('LLEN', KEYS[2]) >= tonumber(ARGV[1]) then
  return 0
end
redis.call('HSET', KEYS[1], unpack(ARGV, 3))
redis.call('LPUSH', KEYS[2], ARGV[2])
return 1
"""

# KEYS: job, segments, heartbeat, processing.
# ARGV: to_status, ttl_seconds, job_id, attempt ('' = any), n_allowed, allowed..., field/values.
# Moves the job to ``to_status`` only if its current status is one of the allowed ones (and, for
# worker writes, the attempt still matches). Entering ``processing`` increments the attempt and
# returns it; entering a terminal status also expires the job, drops the heartbeat and acks the
# queue. Returns 0 when the transition is rejected.
_TRANSITION = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return 0 end
if ARGV[4] ~= '' and redis.call('HGET', KEYS[1], 'attempts') ~= ARGV[4] then return 0 end
local n_allowed = tonumber(ARGV[5])
local ok = false
for i = 1, n_allowed do
  if status == ARGV[5 + i] then ok = true end
end
if not ok then return 0 end
redis.call('HSET', KEYS[1], 'status', ARGV[1])
local result = 1
if ARGV[1] == 'processing' then result = redis.call('HINCRBY', KEYS[1], 'attempts', 1) end
local first_field = 6 + n_allowed
if #ARGV >= first_field then
  redis.call('HSET', KEYS[1], unpack(ARGV, first_field))
end
if ARGV[1] == 'completed' or ARGV[1] == 'failed' or ARGV[1] == 'cancelled' then
  local ttl = tonumber(ARGV[2])
  redis.call('EXPIRE', KEYS[1], ttl)
  redis.call('EXPIRE', KEYS[2], ttl)
  redis.call('DEL', KEYS[3])
  redis.call('LREM', KEYS[4], 0, ARGV[3])
end
return result
"""

# KEYS: job, heartbeat.  ARGV: now, heartbeat_ttl_ms.
# Marks a just-claimed job as owned. A job deleted while queued is left alone (no stray hash).
_MARK_CLAIMED = """
redis.call('SET', KEYS[2], '1', 'PX', tonumber(ARGV[2]))
if redis.call('EXISTS', KEYS[1]) == 1 then
  redis.call('HSET', KEYS[1], 'claimed_at', ARGV[1])
end
return 1
"""

# KEYS: job, segments.  ARGV: segment_json, progress, attempt.
# Appends only while the job is still processing *this* attempt; returns 0 once it was cancelled
# or handed to another worker, which tells the worker to stop.
_APPEND_SEGMENT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then return 0 end
if redis.call('HGET', KEYS[1], 'attempts') ~= ARGV[3] then return 0 end
redis.call('RPUSH', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[1], 'progress', ARGV[2])
return 1
"""

# KEYS: pending, processing.
# ARGV: now, heartbeat_ttl, max_attempts, ttl_seconds, worker_lost_error_json, completed_at_iso.
# For every claimed job whose heartbeat has expired: re-queue it, or fail it after max attempts.
# A job without ``claimed_at`` may have been claimed a moment ago (BLMOVE and the follow-up
# HSET are not atomic), so the sweeper only starts its clock and decides on a later pass.
_RECOVER = """
local now = tonumber(ARGV[1])
local hb_ttl = tonumber(ARGV[2])
local requeued, failed, dropped = 0, 0, 0
for _, id in ipairs(redis.call('LRANGE', KEYS[2], 0, -1)) do
  local jkey = 'job:' .. id
  if redis.call('EXISTS', jkey .. ':heartbeat') == 0 then
    local status = redis.call('HGET', jkey, 'status')
    local claimed = redis.call('HGET', jkey, 'claimed_at')
    if not status or (status ~= 'queued' and status ~= 'processing') then
      redis.call('LREM', KEYS[2], 0, id)
      dropped = dropped + 1
    elseif not claimed then
      redis.call('HSET', jkey, 'claimed_at', ARGV[1])
    elseif now - tonumber(claimed) > hb_ttl then
      redis.call('LREM', KEYS[2], 0, id)
      local attempts = tonumber(redis.call('HGET', jkey, 'attempts') or '0')
      if attempts >= tonumber(ARGV[3]) then
        redis.call('HSET', jkey, 'status', 'failed', 'error', ARGV[5], 'completed_at', ARGV[6])
        redis.call('EXPIRE', jkey, tonumber(ARGV[4]))
        redis.call('EXPIRE', jkey .. ':segments', tonumber(ARGV[4]))
        failed = failed + 1
      else
        redis.call('HSET', jkey, 'status', 'queued', 'progress', '0')
        redis.call('HDEL', jkey, 'claimed_at', 'started_at')
        redis.call('DEL', jkey .. ':segments')
        redis.call('RPUSH', KEYS[1], id)
        requeued = requeued + 1
      end
    end
  end
end
return {requeued, failed, dropped}
"""


def _dump(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "1" if value else "0"
    if hasattr(value, "model_dump_json"):
        return cast(str, value.model_dump_json())
    return str(value)


def _fields(**values: Any) -> list[str]:
    flat: list[str] = []
    for key, value in values.items():
        if value is not None:
            flat += [key, _dump(value)]
    return flat


def _parse_job(raw: Mapping[str, str]) -> Job:
    return Job(
        id=raw["id"],
        status=JobStatus(raw["status"]),
        filename=raw["filename"],
        language=raw.get("language") or None,
        word_timestamps=raw.get("word_timestamps") == "1",
        progress=float(raw.get("progress", 0)),
        attempts=int(raw.get("attempts", 0)),
        created_at=datetime.fromisoformat(raw["created_at"]),
        started_at=datetime.fromisoformat(raw["started_at"]) if raw.get("started_at") else None,
        completed_at=(
            datetime.fromisoformat(raw["completed_at"]) if raw.get("completed_at") else None
        ),
        claimed_at=float(raw["claimed_at"]) if raw.get("claimed_at") else None,
        error=JobErrorInfo.model_validate_json(raw["error"]) if raw.get("error") else None,
        summary=(
            TranscriptionSummary.model_validate_json(raw["summary"]) if raw.get("summary") else None
        ),
    )


class JobStore:
    def __init__(self, redis: Redis, *, result_ttl_seconds: int, max_queue_size: int):
        self._redis = redis
        self._ttl = result_ttl_seconds
        self._max_queue_size = max_queue_size
        self._create_script = redis.register_script(_CREATE_AND_ENQUEUE)
        self._transition_script = redis.register_script(_TRANSITION)
        self._append_script = redis.register_script(_APPEND_SEGMENT)
        self._mark_claimed_script = redis.register_script(_MARK_CLAIMED)
        self._recover_script = redis.register_script(_RECOVER)

    @property
    def redis(self) -> Redis:
        return self._redis

    # ---- Creation & reads ------------------------------------------------------------------

    async def queue_length(self) -> int:
        return int(await self._redis.llen(PENDING_QUEUE))

    async def is_queue_full(self) -> bool:
        return await self.queue_length() >= self._max_queue_size

    async def create(self, job: Job) -> None:
        """Persist a new job and enqueue it atomically. Raises QueueFullError on backpressure."""
        fields = _fields(**job.model_dump(exclude_none=True, exclude={"error", "summary"}))
        created = await self._create_script(
            keys=[job_key(job.id), PENDING_QUEUE],
            args=[self._max_queue_size, job.id, *fields],
        )
        if not created:
            raise QueueFullError(headers={"Retry-After": "30"})

    async def get(self, job_id: str) -> Job | None:
        raw = await cast(Awaitable[dict[str, str]], self._redis.hgetall(job_key(job_id)))
        return _parse_job(raw) if raw.get("status") else None

    async def get_segments(self, job_id: str, start: int = 0) -> list[Segment]:
        items = await cast(
            Awaitable[list[str]], self._redis.lrange(segments_key(job_id), start, -1)
        )
        return [Segment.model_validate_json(item) for item in items]

    async def delete(self, job_id: str) -> None:
        await self._redis.delete(job_key(job_id), segments_key(job_id), heartbeat_key(job_id))

    # ---- Queue -----------------------------------------------------------------------------

    async def claim(self, timeout: int, heartbeat_ttl: float) -> str | None:
        """Block until a job is available and move it to the processing list."""
        job_id = await cast(
            Awaitable[str | None],
            self._redis.blmove(PENDING_QUEUE, PROCESSING_QUEUE, timeout, "RIGHT", "LEFT"),
        )
        if job_id is None:
            return None
        await self._mark_claimed_script(
            keys=[job_key(job_id), heartbeat_key(job_id)],
            args=[time.time(), int(heartbeat_ttl * 1000)],
        )
        return job_id

    async def ack(self, job_id: str) -> None:
        """Remove a job id from the processing list (used when a claimed job is skipped)."""
        await self._redis.lrem(PROCESSING_QUEUE, 0, job_id)
        await self._redis.delete(heartbeat_key(job_id))

    async def remove_from_pending(self, job_id: str) -> None:
        await self._redis.lrem(PENDING_QUEUE, 0, job_id)

    async def heartbeat(self, job_id: str, ttl: float) -> None:
        await self._redis.set(heartbeat_key(job_id), "1", px=int(ttl * 1000))

    # ---- Transitions -----------------------------------------------------------------------

    async def _transition(
        self,
        job_id: str,
        to: JobStatus,
        allowed: tuple[JobStatus, ...],
        *,
        attempt: int | None = None,
        **fields: Any,
    ) -> int:
        result = await self._transition_script(
            keys=[job_key(job_id), segments_key(job_id), heartbeat_key(job_id), PROCESSING_QUEUE],
            args=[
                to.value,
                self._ttl,
                job_id,
                "" if attempt is None else attempt,
                len(allowed),
                *allowed,
                *_fields(**fields),
            ],
        )
        return int(result)

    async def start(self, job_id: str) -> int | None:
        """queued -> processing. Returns the new attempt number (the fencing token every later
        worker write must present), or None if the job is no longer queued."""
        attempt = await self._transition(
            job_id, JobStatus.PROCESSING, (JobStatus.QUEUED,), started_at=utc_now(), progress=0
        )
        return attempt or None

    async def append_segment(
        self, job_id: str, attempt: int, segment: Segment, progress: float
    ) -> bool:
        result = await self._append_script(
            keys=[job_key(job_id), segments_key(job_id)],
            args=[segment.model_dump_json(), f"{progress:.4f}", attempt],
        )
        return bool(result)

    async def complete(self, job_id: str, attempt: int, summary: TranscriptionSummary) -> bool:
        return bool(
            await self._transition(
                job_id,
                JobStatus.COMPLETED,
                (JobStatus.PROCESSING,),
                attempt=attempt,
                summary=summary,
                progress=1,
                completed_at=utc_now(),
            )
        )

    async def fail(self, job_id: str, attempt: int, code: str, message: str) -> bool:
        return bool(
            await self._transition(
                job_id,
                JobStatus.FAILED,
                (JobStatus.PROCESSING,),
                attempt=attempt,
                error=JobErrorInfo(code=code, message=message),
                completed_at=utc_now(),
            )
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or processing job (user action, so not fenced by attempt)."""
        return bool(
            await self._transition(
                job_id,
                JobStatus.CANCELLED,
                (JobStatus.QUEUED, JobStatus.PROCESSING),
                completed_at=utc_now(),
            )
        )

    # ---- Recovery & liveness -----------------------------------------------------------------

    async def recover_orphans(self, *, heartbeat_ttl: float, max_attempts: int) -> tuple[int, ...]:
        """Re-queue or fail jobs whose worker stopped heart-beating. Returns (requeued, failed,
        dropped)."""
        error = JobErrorInfo(
            code=ErrorCode.WORKER_LOST,
            message="The worker processing this job stopped responding.",
        )
        result = await self._recover_script(
            keys=[PENDING_QUEUE, PROCESSING_QUEUE],
            args=[
                time.time(),
                heartbeat_ttl,
                max_attempts,
                self._ttl,
                error.model_dump_json(),
                utc_now().isoformat(),
            ],
        )
        return tuple(int(n) for n in result)

    async def register_worker(self, worker_id: str, ttl: float) -> None:
        await self._redis.set(f"{WORKER_PREFIX}{worker_id}", "1", px=int(ttl * 1000))

    async def unregister_worker(self, worker_id: str) -> None:
        await self._redis.delete(f"{WORKER_PREFIX}{worker_id}")

    async def is_worker_alive(self, worker_id: str) -> bool:
        return bool(await self._redis.exists(f"{WORKER_PREFIX}{worker_id}"))

    async def count_live_workers(self) -> int:
        count = 0
        async for _ in self._redis.scan_iter(match=f"{WORKER_PREFIX}*", count=100):
            count += 1
        return count
