from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MediaFiles
from transcriber.core.errors import DecodeError, InvalidMediaError
from transcriber.media.decode import SAMPLE_RATE, decode_to_wav, load_wav
from transcriber.media.probe import probe


async def test_probe_audio(media: MediaFiles) -> None:
    info = await probe(media.tone, timeout=10)
    assert info.has_audio and not info.has_video
    assert info.duration == pytest.approx(2.0, abs=0.05)


async def test_probe_video(media: MediaFiles) -> None:
    with_audio = await probe(media.video_with_audio, timeout=10)
    assert with_audio.has_audio and with_audio.has_video
    no_audio = await probe(media.video_no_audio, timeout=10)
    assert no_audio.has_video and not no_audio.has_audio


async def test_probe_rejects_garbage(media: MediaFiles) -> None:
    with pytest.raises(InvalidMediaError):
        await probe(media.corrupt, timeout=10)


@pytest.mark.parametrize("name", ["tone", "stereo", "video_with_audio", "speech"])
async def test_decode_produces_16k_mono(media: MediaFiles, tmp_path: Path, name: str) -> None:
    target = tmp_path / "out.wav"
    await decode_to_wav(getattr(media, name), target, timeout=30)
    audio = load_wav(target)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) > SAMPLE_RATE * 0.9
    assert float(np.abs(audio).max()) <= 1.0


async def test_decode_failure(media: MediaFiles, tmp_path: Path) -> None:
    with pytest.raises(DecodeError):
        await decode_to_wav(media.corrupt, tmp_path / "out.wav", timeout=30)
