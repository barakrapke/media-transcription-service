"""faster-whisper (CTranslate2) implementation of the Transcriber protocol.

Model weights are downloaded from the Hugging Face Hub on first use and cached in ``HF_HOME``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from transcriber.core.config import Settings
from transcriber.engine.base import TranscriptionInfo
from transcriber.jobs.models import Segment, Word

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(device: str, requested: str | None) -> str:
    if requested:
        return requested
    return "float16" if device == "cuda" else "int8"


def to_segment(index: int, raw: Any) -> Segment:
    words = None
    if raw.words is not None:
        words = [
            Word(
                start=round(w.start, 3),
                end=round(w.end, 3),
                word=w.word,
                probability=round(w.probability, 4),
            )
            for w in raw.words
        ]
    return Segment(
        id=index,
        start=round(raw.start, 3),
        end=round(raw.end, 3),
        text=raw.text.strip(),
        words=words,
    )


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        self._model_name = settings.model_name
        self._beam_size = settings.beam_size
        device = resolve_device(settings.device)
        compute_type = resolve_compute_type(device, settings.compute_type)

        logger.info(
            "Loading model %s (device=%s, compute_type=%s, cpu_threads=%s)",
            self._model_name,
            device,
            compute_type,
            settings.cpu_threads or "auto",
        )
        started = time.monotonic()
        self._model = WhisperModel(
            self._model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=settings.cpu_threads,
        )
        logger.info("Model loaded in %.1fs", time.monotonic() - started)

    @property
    def model_name(self) -> str:
        return self._model_name

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        raw_segments, raw_info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._beam_size,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        info = TranscriptionInfo(
            language=raw_info.language,
            language_probability=round(raw_info.language_probability, 4),
            duration=round(raw_info.duration, 3),
        )
        segments = (to_segment(i, raw) for i, raw in enumerate(raw_segments))
        return info, segments

    def warmup(self) -> None:
        """Run one short inference so lazy initialisation (kernels, VAD model) happens at startup
        rather than on the first real job."""
        started = time.monotonic()
        noise = (np.random.default_rng(0).standard_normal(16_000 * 2) * 0.01).astype(np.float32)
        segments, _ = self._model.transcribe(noise, language="en", beam_size=1, vad_filter=False)
        for _ in segments:
            pass
        _, _ = self._model.transcribe(noise, language="en", beam_size=1, vad_filter=True)
        logger.info("Model warm-up finished in %.1fs", time.monotonic() - started)
