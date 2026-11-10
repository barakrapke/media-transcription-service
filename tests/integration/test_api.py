from pathlib import Path

import httpx
import pytest

from tests.conftest import FakeTranscriber, MediaFiles
from transcriber.core.config import Settings
from transcriber.jobs.models import JobStatus
from transcriber.jobs.store import JobStore
from transcriber.media.storage import UploadStorage
from transcriber.worker.processor import JobProcessor

API = "/v1/transcriptions"


async def upload(client: httpx.AsyncClient, path: Path, **data: str) -> httpx.Response:
    with path.open("rb") as fh:
        return await client.post(API, files={"file": (path.name, fh)}, data=data)


async def run_worker_once(
    store: JobStore, storage: UploadStorage, transcriber: FakeTranscriber, settings: Settings
) -> None:
    job_id = await store.claim(timeout=1, heartbeat_ttl=5)
    assert job_id is not None
    await JobProcessor(store, storage, transcriber, settings).process(job_id)


@pytest.fixture
async def completed_job(
    client: httpx.AsyncClient, store: JobStore, storage: UploadStorage,
    transcriber: FakeTranscriber, settings: Settings, media: MediaFiles,
) -> str:  # fmt: skip
    response = await upload(client, media.tone)
    await run_worker_once(store, storage, transcriber, settings)
    return str(response.json()["id"])


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()["error"]
    assert body["code"] == code
    assert body["message"]
    assert body["request_id"] == response.headers["x-request-id"]


# ---- Upload ---------------------------------------------------------------------------------


async def test_upload_creates_queued_job(
    client: httpx.AsyncClient, store: JobStore, storage: UploadStorage, media: MediaFiles
) -> None:
    response = await upload(client, media.video_with_audio, language="EN", word_timestamps="true")
    assert response.status_code == 202, response.text
    body = response.json()
    assert response.headers["location"] == f"{API}/{body['id']}"
    assert body["status"] == "queued"
    assert body["filename"] == "with_audio.mp4"
    assert body["language"] == "en"
    assert body["word_timestamps"] is True
    assert body["result"] is None
    assert await store.queue_length() == 1
    assert storage.path_for(body["id"]).exists()


@pytest.mark.parametrize(
    ("fixture", "data", "status", "code"),
    [
        ("video_no_audio", {}, 422, "NO_AUDIO_STREAM"),
        ("corrupt", {}, 422, "INVALID_MEDIA"),
        ("tone", {"language": "xx"}, 422, "UNSUPPORTED_LANGUAGE"),
    ],
)
async def test_upload_rejections(
    client: httpx.AsyncClient, storage: UploadStorage, media: MediaFiles,
    fixture: str, data: dict[str, str], status: int, code: str,
) -> None:  # fmt: skip
    response = await upload(client, getattr(media, fixture), **data)
    assert_error(response, status, code)
    assert list(storage.list_older_than(-1)) == []  # rejected uploads are not kept


async def test_upload_too_long(
    client: httpx.AsyncClient, settings: Settings, redis: object, media: MediaFiles
) -> None:
    from transcriber.api.app import create_app

    app = create_app(settings.model_copy(update={"max_duration_seconds": 1}), redis=redis)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as short_client:
        assert_error(await upload(short_client, media.tone), 422, "DURATION_TOO_LONG")


async def test_upload_too_large_is_rejected_while_streaming(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (7 * 1024 * 1024))  # limit is 5 MB (+1 MB multipart allowance)
    assert_error(await upload(client, big), 413, "FILE_TOO_LARGE")


async def test_upload_too_large_without_content_length(client: httpx.AsyncClient) -> None:
    async def chunked_body():  # type: ignore[no-untyped-def]
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n\r\n'
        for _ in range(8):
            yield b"\0" * (1024 * 1024)
        yield b"\r\n--b--\r\n"

    response = await client.post(
        API, content=chunked_body(), headers={"content-type": "multipart/form-data; boundary=b"}
    )
    assert_error(response, 413, "FILE_TOO_LARGE")


async def test_upload_file_larger_than_limit_but_within_allowance(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (5 * 1024 * 1024 + 10))
    assert_error(await upload(client, big), 413, "FILE_TOO_LARGE")


async def test_upload_requires_file(client: httpx.AsyncClient) -> None:
    response = await client.post(API, data={"language": "en"})
    assert_error(response, 422, "VALIDATION_ERROR")
    assert response.json()["error"]["details"][0]["field"] == "file"


async def test_queue_full(
    client: httpx.AsyncClient, store: JobStore, settings: Settings, media: MediaFiles
) -> None:
    for _ in range(settings.max_queue_size):
        assert (await upload(client, media.tone)).status_code == 202
    response = await upload(client, media.tone)
    assert_error(response, 503, "QUEUE_FULL")
    assert response.headers["retry-after"] == "30"


# ---- Read & export ----------------------------------------------------------------------------


async def test_get_completed_job(client: httpx.AsyncClient, completed_job: str) -> None:
    response = await client.get(f"{API}/{completed_job}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["progress"] == 1
    assert body["result"]["text"] == "Hello world. This is a test."
    assert body["result"]["language"] == "en"
    assert [s["text"] for s in body["result"]["segments"]] == ["Hello world.", "This is a test."]


@pytest.mark.parametrize(
    ("fmt", "media_type", "first_line"),
    [
        ("txt", "text/plain", "Hello world."),
        ("srt", "application/x-subrip", "1"),
        ("vtt", "text/vtt", "WEBVTT"),
    ],
)
async def test_export(
    client: httpx.AsyncClient, completed_job: str, fmt: str, media_type: str, first_line: str
) -> None:
    response = await client.get(f"{API}/{completed_job}", params={"format": fmt})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.headers["content-disposition"].startswith(f'attachment; filename="tone.{fmt}"')
    assert response.text.splitlines()[0] == first_line


async def test_export_before_completion(client: httpx.AsyncClient, media: MediaFiles) -> None:
    job_id = (await upload(client, media.tone)).json()["id"]
    assert_error(await client.get(f"{API}/{job_id}?format=srt"), 409, "JOB_NOT_COMPLETED")


async def test_export_unknown_format(client: httpx.AsyncClient, completed_job: str) -> None:
    assert_error(await client.get(f"{API}/{completed_job}?format=pdf"), 422, "VALIDATION_ERROR")


@pytest.mark.parametrize("job_id", ["0" * 32, "not-a-job", "..%2E%2E"])
async def test_unknown_job(client: httpx.AsyncClient, job_id: str) -> None:
    assert_error(await client.get(f"{API}/{job_id}"), 404, "JOB_NOT_FOUND")
    assert_error(await client.delete(f"{API}/{job_id}"), 404, "JOB_NOT_FOUND")
    assert_error(await client.get(f"{API}/{job_id}/events"), 404, "JOB_NOT_FOUND")


# ---- Delete -----------------------------------------------------------------------------------


async def test_delete_queued_job_cancels_it(
    client: httpx.AsyncClient, store: JobStore, storage: UploadStorage, media: MediaFiles
) -> None:
    job_id = (await upload(client, media.tone)).json()["id"]
    response = await client.delete(f"{API}/{job_id}")
    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert await store.queue_length() == 0
    assert not storage.path_for(job_id).exists()
    # A second DELETE removes the cancelled job entirely.
    assert (await client.delete(f"{API}/{job_id}")).status_code == 204
    assert (await client.get(f"{API}/{job_id}")).status_code == 404


async def test_delete_processing_job_cancels_it(
    client: httpx.AsyncClient, store: JobStore, media: MediaFiles
) -> None:
    job_id = (await upload(client, media.tone)).json()["id"]
    await store.claim(timeout=1, heartbeat_ttl=5)
    await store.start(job_id)
    response = await client.delete(f"{API}/{job_id}")
    assert response.status_code == 202
    job = await store.get(job_id)
    assert job is not None and job.status is JobStatus.CANCELLED


async def test_delete_completed_job(client: httpx.AsyncClient, completed_job: str) -> None:
    assert (await client.delete(f"{API}/{completed_job}")).status_code == 204
    assert (await client.get(f"{API}/{completed_job}")).status_code == 404


# ---- Health, docs, UI, errors -------------------------------------------------------------------


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.json() == {"status": "ok"}


async def test_readyz_requires_a_worker(client: httpx.AsyncClient, store: JobStore) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["workers"] == 0
    await store.register_worker("w1", ttl=5)
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "redis": True, "workers": 1, "queue_length": 0}


async def test_redis_outage_returns_503(settings: Settings) -> None:
    from redis.asyncio import Redis

    from transcriber.api.app import create_app

    unreachable = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    app = create_app(settings, redis=unreachable)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as http:
        response = await http.get(f"{API}/{'0' * 32}")
        assert_error(response, 503, "DEPENDENCY_UNAVAILABLE")
        ready = await http.get("/readyz")
        assert ready.status_code == 503 and ready.json()["redis"] is False
    await unreachable.aclose()


async def test_request_id_is_propagated(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"
    generated = await client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["x-request-id"] != "bad id with spaces"


async def test_unknown_route_and_method(client: httpx.AsyncClient) -> None:
    assert_error(await client.get("/v1/nothing"), 404, "NOT_FOUND")
    assert_error(await client.put(API), 405, "METHOD_NOT_ALLOWED")


async def test_ui_and_docs_are_served(client: httpx.AsyncClient) -> None:
    ui = await client.get("/")
    assert ui.status_code == 200 and "text/html" in ui.headers["content-type"]
    assert (await client.get("/docs")).status_code == 200
    schema = (await client.get("/openapi.json")).json()
    assert f"{API}/{{job_id}}/events" in schema["paths"]
