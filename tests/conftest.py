"""Shared fixtures: settings, in-memory Redis, a fake transcriber and generated media files."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import fakeredis
import httpx
import numpy as np
import numpy.typing as npt
import pytest

from transcriber.api.app import create_app
from transcriber.core.config import Settings
from transcriber.engine.base import TranscriptionInfo
from transcriber.jobs.models import Segment
from transcriber.jobs.store import JobStore
from transcriber.media.storage import UploadStorage

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---- Settings / Redis / store ---------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        max_upload_mb=5,
        max_duration_seconds=60,
        max_queue_size=10,
        job_timeout_seconds=30,
        decode_timeout_seconds=30,
        heartbeat_interval_seconds=0.05,
        heartbeat_ttl_seconds=0.2,
        sweep_interval_seconds=0.05,
        sse_poll_interval_seconds=0.01,
        worker_id="test-worker",
        model_warmup=False,
    )


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(decode_responses=True)

    # fakeredis implements blocking commands by blocking the event loop itself, which would
    # starve the worker's other tasks. Real Redis doesn't; emulate it with a yielding LMOVE.
    async def blmove(src: str, dst: str, timeout: float, wherefrom: str, whereto: str) -> object:
        moved = await client.lmove(src, dst, wherefrom, whereto)
        if moved is None:
            await asyncio.sleep(0.01)
        return moved

    client.blmove = blmove  # type: ignore[method-assign]
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
def store(redis: fakeredis.FakeAsyncRedis, settings: Settings) -> JobStore:
    return JobStore(
        redis,
        result_ttl_seconds=settings.result_ttl_seconds,
        max_queue_size=settings.max_queue_size,
    )


@pytest.fixture
def storage(settings: Settings) -> UploadStorage:
    s = UploadStorage(settings.upload_dir)
    s.ensure_root()
    return s


@pytest.fixture
async def client(
    settings: Settings, redis: fakeredis.FakeAsyncRedis
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, redis=redis)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


# ---- Fake transcriber -----------------------------------------------------------------------


@dataclass
class FakeTranscriber:
    """Deterministic stand-in for the model. Behaviour is configured per test."""

    segments: list[Segment] = field(
        default_factory=lambda: [
            Segment(id=0, start=0.0, end=1.0, text="Hello world."),
            Segment(id=1, start=1.0, end=2.0, text="This is a test."),
        ]
    )
    language: str = "en"
    duration: float = 2.0
    delay_per_segment: float = 0.0
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)
    model_name: str = "fake-model"
    started: threading.Event = field(default_factory=threading.Event)

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        self.calls.append(
            {"samples": len(audio), "language": language, "word_timestamps": word_timestamps}
        )
        self.started.set()
        if self.error is not None:
            raise self.error
        info = TranscriptionInfo(
            language=language or self.language, language_probability=0.99, duration=self.duration
        )

        def generate() -> Iterator[Segment]:
            for segment in self.segments:
                if self.delay_per_segment:
                    time.sleep(self.delay_per_segment)
                yield segment

        return info, generate()

    def warmup(self) -> None:
        pass


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


# ---- Media files (generated with ffmpeg once per session) ------------------------------------


def _ffmpeg_works() -> bool:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


@dataclass(frozen=True)
class MediaFiles:
    tone: Path  # 2 s mono 440 Hz WAV
    stereo: Path  # 1 s stereo WAV
    video_with_audio: Path  # 1 s MP4 with an AAC track
    video_no_audio: Path  # 1 s MP4 without audio
    corrupt: Path  # random bytes
    speech: Path  # committed TTS clip (see tests/fixtures/README.md)


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> MediaFiles:
    if not _ffmpeg_works():
        pytest.skip("ffmpeg/ffprobe are not installed or not working")
    root = tmp_path_factory.mktemp("media")

    def ffmpeg(*args: str) -> None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)

    files = MediaFiles(
        tone=root / "tone.wav",
        stereo=root / "stereo.wav",
        video_with_audio=root / "with_audio.mp4",
        video_no_audio=root / "no_audio.mp4",
        corrupt=root / "corrupt.mp3",
        speech=FIXTURES_DIR / "speech_en.m4a",
    )
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "44100", str(files.tone))
    ffmpeg(
        "-f", "lavfi", "-i", "aevalsrc=sin(440*2*PI*t)|sin(660*2*PI*t):d=1:s=44100",
        str(files.stereo),
    )  # fmt: skip
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(files.video_with_audio),
    )  # fmt: skip
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-c:v", "mpeg4", "-an", str(files.video_no_audio),
    )  # fmt: skip
    files.corrupt.write_bytes(np.random.default_rng(0).bytes(4096))
    return files
