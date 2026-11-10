"""Transcription engine interface.

The worker depends only on this protocol, so the engine can be swapped (e.g. for a
``transformers`` pipeline) without touching the job pipeline, and tests can use a fake.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from transcriber.jobs.models import Segment


@dataclass(frozen=True)
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float


class Transcriber(Protocol):
    @property
    def model_name(self) -> str: ...

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        """Start transcribing. Segments are produced lazily by the returned iterator."""
        ...

    def warmup(self) -> None: ...
