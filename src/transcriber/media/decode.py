"""Decoding media to the format the model expects: 16 kHz, mono, PCM."""

from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from transcriber.core.errors import DecodeError, DecodeTimeoutError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


async def decode_to_wav(source: Path, target: Path, timeout: float) -> None:
    """Decode any audio/video file to a 16 kHz mono 16-bit WAV (multi-channel is downmixed)."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v", "error",
        "-i", str(source),
        "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise DecodeTimeoutError() from None

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()
        logger.warning("ffmpeg failed (exit %s): %s", process.returncode, detail[-3:])
        raise DecodeError()


def load_wav(path: Path) -> npt.NDArray[np.float32]:
    """Read a 16-bit mono WAV produced by ``decode_to_wav`` as float32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise DecodeError("Unexpected intermediate audio format.")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
