"""Media inspection with ffprobe.

Whether a file is accepted is decided by what ffprobe can actually read - not by the file
extension or the client-supplied Content-Type, which are both untrusted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcriber.core.errors import InvalidMediaError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaInfo:
    format_name: str
    duration: float | None
    has_audio: bool
    has_video: bool


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def parse_probe_output(data: dict[str, Any]) -> MediaInfo:
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    codec_types = {stream.get("codec_type") for stream in streams}

    duration = _to_float(fmt.get("duration"))
    if duration is None:  # some containers only report per-stream durations
        stream_durations = [_to_float(s.get("duration")) for s in streams]
        known = [d for d in stream_durations if d is not None]
        duration = max(known) if known else None

    return MediaInfo(
        format_name=str(fmt.get("format_name", "unknown")),
        duration=duration,
        has_audio="audio" in codec_types,
        has_video="video" in codec_types,
    )


async def probe(path: Path, timeout: float) -> MediaInfo:
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise InvalidMediaError("Inspecting the file took too long.") from None

    if process.returncode != 0:
        logger.info("ffprobe rejected file: %s", stderr.decode(errors="replace").strip()[-500:])
        raise InvalidMediaError()
    try:
        return parse_probe_output(json.loads(stdout))
    except (json.JSONDecodeError, AttributeError) as exc:
        raise InvalidMediaError() from exc
