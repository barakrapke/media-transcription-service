# Architecture

This document describes how the Media Transcription Service is built and, more importantly, **why** —
every significant decision is listed with the alternatives that were considered and the trade-off
that was accepted.

---

## 1. Overview

A user uploads an audio or video file and receives a timestamped transcription. Transcription is
slow relative to an HTTP request (minutes for long files on CPU), so the service is built around
**asynchronous jobs**: the upload returns immediately with a job id, a worker transcribes in the
background, and the client follows progress by polling or via Server-Sent Events (SSE) — including
segments streamed live as the model produces them.

```
 Browser UI / curl
        │  POST /v1/transcriptions              GET /v1/transcriptions/{id}[?format=…]
        │                                       GET /v1/transcriptions/{id}/events  (SSE)
        ▼
 ┌────────────────┐  enqueue job id   ┌─────────┐   BLMOVE   ┌────────────────────────────────┐
 │ API  (FastAPI) │ ────────────────▶ │  Redis  │ ◀───────── │ Worker  (×N)                   │
 │ • validate     │                   │ • queue │            │ • ffmpeg → 16 kHz mono WAV     │
 │ • store upload │ ◀── job state ─── │ • state │ ── state ─▶│ • faster-whisper (loaded once) │
 │ • status / SSE │                   │ • TTLs  │            │ • progress + segments → Redis  │
 └───────┬────────┘                   └─────────┘            └───────────────┬────────────────┘
         └──────────────── shared volume: /data/uploads ─────────────────────┘
```

| Component | Responsibility | Scales by |
|---|---|---|
| **API** | Upload streaming, validation (size, ffprobe), job creation, status, export, SSE, static UI | Replicas (stateless) |
| **Worker** | Decode, transcribe, publish progress/segments, heartbeat, recovery sweep | Replicas — one job per worker |
| **Redis** | Job queue, job state, partial segments, heartbeats, TTL-based expiry | Single instance (AOF on) |
| **Shared volume** | Hand-off of uploaded media from API to worker | — |

**Dependency rule:** `api` and `worker` never import each other. Both depend only on the shared
domain packages (`core`, `jobs`, `media`, `engine`, `formatters`).

---

## 2. Request flow

1. **Upload** — `POST /v1/transcriptions` (multipart: `file`, optional `language`, `word_timestamps`).
   - The size limit is enforced while the body streams in: an ASGI middleware rejects a declared
     `Content-Length` above the limit immediately, and otherwise counts bytes and answers `413` as
     soon as `MAX_UPLOAD_MB` is crossed (also for chunked uploads without a length).
   - The file is copied to `/data/uploads/{job_id}` in chunks (the size is checked again there).
     The user's filename is stored only as metadata.
   - `ffprobe` verifies the file is readable, has ≥1 audio stream and is within `MAX_DURATION_SECONDS`.
   - Queue length is checked against `MAX_QUEUE_SIZE` (backpressure → `503` + `Retry-After`) —
     cheaply before saving the file, and again atomically when enqueuing.
   - One Lua script creates the job hash in state `queued` and pushes the id to the pending queue.
   - Response: `202 Accepted` with the job resource and a `Location` header.
2. **Processing** — a worker `BLMOVE`s the job id from `pending` to `processing`, sets its heartbeat
   and `claimed_at`, transitions the job to `processing`, decodes with ffmpeg, then runs the model
   in a thread. The thread hands each segment to the event loop, which appends it to the segment
   list and updates `progress` — a write that succeeds only while the job is still `processing`,
   so cancellation is noticed at the next segment. The job deadline is checked between segments;
   the heartbeat is refreshed by a separate task.
3. **Completion** — the full result is written, the job transitions to `completed`, files are
   deleted, the job id is removed from `processing`, and the job hash gets its TTL.
4. **Reading** — clients poll `GET /v1/transcriptions/{id}` or subscribe to `/events`. Completed
   jobs can be exported with `?format=txt|srt|vtt`.

---

## 3. Job lifecycle

```
queued ──▶ processing ──▶ completed
   │            ├────────▶ failed
   └────────────┴────────▶ cancelled
```

- Transitions are performed by a Lua script that checks the current state, so they are atomic.
  Terminal states (`completed`, `failed`, `cancelled`) are immutable — a late or zombie worker
  cannot overwrite them.
- Job state expires `RESULT_TTL_SECONDS` (default 24 h) after reaching a terminal state.

### Redis data model

| Key | Type | Content |
|---|---|---|
| `job:{id}` | hash | `status, filename, language, word_timestamps, progress, attempts, created_at, started_at, completed_at, claimed_at, error (JSON), summary (JSON)` — the result without segments |
| `job:{id}:segments` | list | Segment JSON, appended as produced. The single source of segments: used for the SSE live view and replay, and to assemble the final result |
| `job:{id}:heartbeat` | string, TTL | Refreshed by the worker owning the job |
| `queue:pending` | list | Job ids waiting for a worker |
| `queue:processing` | list | Job ids claimed by a worker |
| `worker:{worker_id}` | string, TTL | Worker liveness (used by `/readyz` and worker healthcheck) |

---

## 4. Reliability

| Concern | Mechanism |
|---|---|
| Worker crashes mid-job | Reliable-queue pattern: `BLMOVE pending → processing`. A **recovery sweeper** (runs in every worker, every 30 s) finds ids in `processing` whose heartbeat has expired and whose `claimed_at` is older than the heartbeat TTL (avoids racing a worker that has just claimed the job). If `attempts < MAX_ATTEMPTS` the job is re-queued, otherwise it fails with `WORKER_LOST`. |
| Retry policy | Only infrastructure failures (lost worker) are retried. Deterministic failures (`DECODE_ERROR`, `TRANSCRIPTION_ERROR`) are not — they would fail again. |
| Long / hung jobs | ffmpeg runs with a subprocess timeout (`DECODE_TIMEOUT`). Inference checks the job deadline between segments and stops cleanly (`JOB_TIMEOUT`). |
| Cancellation | `DELETE` moves the job to `cancelled` atomically. A queued job is also removed from the queue (a worker that claims it anyway skips it, because `queued → processing` fails). A processing job is stopped by its worker: its next segment write is rejected, so it stops and cleans up. No separate cancel flag is needed. |
| Overload | `MAX_QUEUE_SIZE` → `503 QUEUE_FULL` with `Retry-After`. |
| Shutdown | Worker on `SIGTERM` stops claiming jobs and finishes the current one within `stop_grace_period`; if killed, the sweeper re-queues it. API closes SSE streams; clients reconnect and get a replay. |
| Redis restart | AOF persistence enabled. |
| Disk leaks | Upload and temporary WAV are deleted in `finally`; the sweeper also removes orphaned files older than the TTL. |

---

## 5. Transcription engine

- **faster-whisper** (CTranslate2) with weights from Hugging Face (`Systran/faster-whisper-*`).
  Default model: **`small`**, `int8` on CPU, `float16` on CUDA. Configurable via `MODEL_NAME`
  (e.g. `large-v3-turbo` for GPU, `ivrit-ai/whisper-large-v3-ct2` for Hebrew).
- Behind a `Transcriber` protocol (`engine/base.py`) so the engine is swappable and tests use a
  fake implementation.
- Device auto-detected; `CPU_THREADS` pins CTranslate2 intra-op threads.
- Decoding: `beam_size=5`, Silero VAD filter (skips silence — faster, fewer hallucinations),
  segment-level timestamps; word-level timestamps opt-in per job.
- Language auto-detected unless provided; detected language and probability are returned.
- The model is downloaded once into a named volume (`HF_HOME`) on the worker's first start,
  loaded once per worker process and warmed up with a dummy inference before the worker
  starts claiming jobs.

---

## 6. Media pipeline

- **Acceptance is decided by `ffprobe`, not by extension or Content-Type** — anything ffmpeg can
  read that contains an audio stream (audio or video containers) is accepted.
- **API side:** streamed upload with a size cap, ffprobe check (readable, has audio, duration).
- **Worker side:** explicit `ffmpeg` subprocess → 16 kHz mono PCM WAV (downmixes multi-channel),
  with a timeout. Decoding errors are distinguishable from model errors. Memory footprint:
  ~230 MB per hour of audio as float32.
- Silence / no speech → `completed` with empty text and no segments (not an error).

---

## 7. API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/transcriptions` | Upload media, create job → `202` |
| `GET` | `/v1/transcriptions/{id}` | Job resource (status, progress, result) |
| `GET` | `/v1/transcriptions/{id}?format=txt\|srt\|vtt` | Export completed transcript as a download (`409` if not completed) |
| `GET` | `/v1/transcriptions/{id}/events` | SSE stream |
| `DELETE` | `/v1/transcriptions/{id}` | Cancel job / delete its data |
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Redis reachable and ≥1 live worker |
| `GET` | `/docs` | OpenAPI / Swagger UI |
| `GET` | `/` | Web UI |

### Job resource

```json
{
  "id": "3f2c…",
  "status": "completed",
  "progress": 1.0,
  "filename": "meeting.mp4",
  "created_at": "…", "started_at": "…", "completed_at": "…",
  "error": null,
  "result": {
    "text": "Hello everyone…",
    "language": "en",
    "language_probability": 0.98,
    "duration": 312.4,
    "model": "Systran/faster-whisper-small",
    "processing_time": 41.2,
    "segments": [{"id": 0, "start": 0.0, "end": 4.2, "text": "Hello everyone…", "words": null}]
  }
}
```

### SSE events

| Event | Payload |
|---|---|
| `status` | `{status}` on every state change |
| `progress` | `{progress}` |
| `segment` | One segment, as soon as the model produces it |
| `completed` | Full result; stream closes |
| `failed` | `{error}`; stream closes |

On connect the server first replays the current state and all segments produced so far, so a late
or reconnecting client never misses data. A terminal `status` is always sent after the final
segments, so it is the last event before `completed`/`failed`. If a job is re-queued after a lost
worker, clients receive `status: queued` and the segments start over. Internally the SSE endpoint reads the job state from
Redis on a short interval and emits deltas — no pub/sub is required.

### Errors

Every error uses one shape:

```json
{"error": {"code": "NO_AUDIO_STREAM", "message": "The file has no audio track", "request_id": "…"}}
```

| Where | Code | HTTP |
|---|---|---|
| API | `VALIDATION_ERROR` | 422 |
| API | `FILE_TOO_LARGE` | 413 |
| API | `INVALID_MEDIA` | 422 |
| API | `NO_AUDIO_STREAM` | 422 |
| API | `DURATION_TOO_LONG` | 422 |
| API | `UNSUPPORTED_LANGUAGE` | 422 |
| API | `JOB_NOT_FOUND` | 404 |
| API | `JOB_NOT_COMPLETED` | 409 |
| API | `QUEUE_FULL` | 503 |
| API | `DEPENDENCY_UNAVAILABLE` | 503 |
| API | `INTERNAL_ERROR` | 500 (details logged, never returned) |
| Job | `DECODE_ERROR`, `DECODE_TIMEOUT`, `TRANSCRIPTION_ERROR`, `JOB_TIMEOUT`, `WORKER_LOST` | in `job.error` |

---

## 8. Web UI

A single static page (vanilla HTML/JS/CSS) served by the API, consuming only the public API:
drag-and-drop upload with upload progress, optional language and word-timestamp options, live
status and segments via SSE, a media player where clicking a segment seeks to it, copy and
TXT/SRT/VTT download, job id kept in the URL (`/#job=…`) so refresh and links work, dark mode.

---

## 9. Configuration, deployment and observability

- **Configuration:** `pydantic-settings`, environment variables only, validated at startup
  (fail fast). All variables documented in `.env.example`.
- **Packaging:** Python 3.12, `uv` with a lockfile, `src/` layout. One image, two commands
  (API / worker).
- **Container:** multi-stage build on `python:3.12-slim` + ffmpeg, non-root user, healthcheck.
- **Compose:** `redis` (AOF), `api` (port 8000), `worker` (scalable with `--scale worker=N`);
  named volumes `uploads`, `model-cache`, `redis-data`.
- **GPU:** `docker-compose.gpu.yml` override (device reservation, `DEVICE=cuda`, cuBLAS/cuDNN
  wheels). **Not tested** — no GPU was available during development.
- **Logging:** structured JSON (text format for local dev); `request_id` on every API log line
  (`X-Request-ID`), `job_id` on every worker log line.
- **Health:** `/healthz`, `/readyz`; worker liveness via a heartbeat key checked by its container
  healthcheck.

---

## 10. Testing strategy

| Layer | Scope | Dependencies |
|---|---|---|
| Unit | Formatters, settings, state transitions, error mapping, ffprobe parsing, progress | none |
| Integration | API with a fake transcriber (upload → result, validation, export, SSE + replay, DELETE); worker lifecycle, timeout, cancel, decode error, crash recovery | fakeredis, ffmpeg |
| E2E | Full Docker Compose stack with the real `tiny` model and a committed speech clip; fuzzy assertion on the transcript | Docker |

Validation fixtures (tone, video without audio, corrupt file, stereo) are generated with ffmpeg at
test time. Coverage gate: 80 % on unit + integration. CI (GitHub Actions): lint (ruff, mypy),
tests with coverage, and E2E.

---

## 11. Decision record

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| D1 | Async jobs, polling + SSE | Synchronous request; async + `?wait=true` | Long files exceed HTTP/proxy timeouts; async decouples upload from compute and survives client disconnects. SSE enables live transcript streaming. |
| D2 | Separate API and worker processes joined by Redis | Single process with background thread | Independent scaling, model isolated from the API, jobs survive restarts. |
| D3 | Custom thin worker on Redis (`BLMOVE` reliable queue) | Celery, RQ, arq, RabbitMQ | Job state must be custom anyway (progress, partial segments for SSE), so a framework would add little beyond queue mechanics. RQ forks per job (bad with a loaded model); Celery is heavy for this scope. |
| D4 | Redis hashes with TTL as job store | Postgres, SQLite | Already in the stack, fast for SSE reads, transcripts are short-lived. Postgres is the production choice for durable history. |
| D5 | faster-whisper with HF weights, behind a `Transcriber` interface | HF `transformers` pipeline; NVIDIA NeMo | ~4× faster on CPU, int8, no torch (smaller image), lazy segment generator fits SSE. Confirmed acceptable by the reviewer. |
| D6 | Default model `small` / int8 | `tiny`, `base`, `large-v3(-turbo)` | Reasonable quality while staying usable on a CPU-only laptop; fully configurable. |
| D7 | `ffprobe` decides acceptance | Extension / MIME allowlist | Extensions and Content-Type are untrusted and incomplete. |
| D8 | Explicit ffmpeg decode in the worker | Let faster-whisper decode via PyAV | Clear separation of decode vs model errors, timeout, explicit downmix. |
| D9 | Result embedded in the job resource; export via `?format=` | Separate `/result` and `/export.srt` endpoints | One resource, simpler client. |
| D10 | Static vanilla UI served by the API | React SPA, Gradio/Streamlit, no UI | No build step or CORS; exercises the public API exactly like any client. |
| D11 | Retry only infrastructure failures | Retry everything | Deterministic failures would fail again and waste worker time. |
| D12 | One image, two commands | Separate API / worker images | Simpler build; slim API image noted as future work. |
| D13 | fakeredis in tests | testcontainers | Tests run without Docker; E2E covers real Redis. |
| D14 | Cancellation through the job status | A separate cancel-flag key | The segment write already checks the status atomically, so a status change stops the worker with no extra key, no extra read, and immediate visibility to clients. |
