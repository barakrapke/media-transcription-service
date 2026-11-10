"""Transcript export formats."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from transcriber.jobs.models import Segment


def _timestamp(seconds: float, separator: str) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def to_srt(segments: Sequence[Segment]) -> str:
    blocks = [
        f"{n}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{s.text}\n"
        for n, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: Sequence[Segment]) -> str:
    blocks = [
        f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{s.text}\n" for s in segments
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(segments: Sequence[Segment]) -> str:
    return "".join(f"{s.text}\n" for s in segments)


ExportFormat = tuple[Callable[[Sequence[Segment]], str], str, str]

# format -> (renderer, media type, file extension)
EXPORT_FORMATS: Final[dict[str, ExportFormat]] = {
    "txt": (to_txt, "text/plain; charset=utf-8", "txt"),
    "srt": (to_srt, "application/x-subrip; charset=utf-8", "srt"),
    "vtt": (to_vtt, "text/vtt; charset=utf-8", "vtt"),
}
