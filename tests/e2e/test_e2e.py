"""End-to-end test against a running stack (real Redis, real worker, real model).

Run with ``make e2e`` (starts a dedicated Compose project with the ``tiny`` model), or point it
at any deployment: ``E2E_BASE_URL=http://host:8000 pytest -m e2e``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
READY_TIMEOUT_SECONDS = float(os.environ.get("E2E_READY_TIMEOUT", "600"))
SPEECH = Path(__file__).parent.parent / "fixtures" / "speech_en.m4a"
EXPECTED_WORDS = {"quick", "brown", "fox", "lazy", "dog", "speech", "recognition", "text"}


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=BASE_URL, timeout=30) as http:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while True:
            try:
                if http.get("/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"{BASE_URL} did not become ready within {READY_TIMEOUT_SECONDS}s")
            time.sleep(2)
        yield http


def stream_events(client: httpx.Client, job_id: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    event = None
    with client.stream("GET", f"/v1/transcriptions/{job_id}/events", timeout=300) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event:
                events.append((event, json.loads(line.removeprefix("data: "))))
    return events


def test_transcribe_speech_end_to_end(client: httpx.Client) -> None:
    with SPEECH.open("rb") as fh:
        response = client.post(
            "/v1/transcriptions",
            files={"file": (SPEECH.name, fh, "audio/mp4")},
            data={"word_timestamps": "true"},
        )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]

    events = stream_events(client, job_id)
    names = [name for name, _ in events]
    assert "segment" in names
    assert names[-1] == "completed", events[-1]

    job = client.get(f"/v1/transcriptions/{job_id}").json()
    assert job["status"] == "completed"
    result = job["result"]
    assert result["language"] == "en"
    assert 5 < result["duration"] < 7
    words = {w.strip(".,!?").lower() for w in result["text"].split()}
    missing = EXPECTED_WORDS - words
    assert len(missing) <= 2, f"transcript {result['text']!r} is missing {missing}"
    assert result["segments"][0]["words"], "word timestamps were requested"

    srt = client.get(f"/v1/transcriptions/{job_id}", params={"format": "srt"})
    assert srt.status_code == 200
    assert "-->" in srt.text

    assert client.delete(f"/v1/transcriptions/{job_id}").status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 404


def test_rejects_non_media(client: httpx.Client) -> None:
    response = client.post(
        "/v1/transcriptions", files={"file": ("notes.txt", b"just some text" * 50, "text/plain")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_MEDIA"
