"""faster-whisper adapter, tested against a stubbed WhisperModel (no model download)."""

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from transcriber.core.config import Settings
from transcriber.engine import faster_whisper as engine


def raw_segment(start: float, end: float, text: str, words: list[Any] | None = None) -> Any:
    return SimpleNamespace(start=start, end=end, text=text, words=words)


class StubWhisperModel:
    instances: ClassVar[list["StubWhisperModel"]] = []

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        StubWhisperModel.instances.append(self)

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)
        info = SimpleNamespace(language="en", language_probability=0.987654, duration=2.0)
        segments = iter([raw_segment(0.0, 1.23456, "  Hello  "), raw_segment(1.3, 2.0, " world")])
        return segments, info


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> type[StubWhisperModel]:
    import faster_whisper

    StubWhisperModel.instances.clear()
    monkeypatch.setattr(faster_whisper, "WhisperModel", StubWhisperModel)
    monkeypatch.setattr(engine, "resolve_device", lambda requested: "cpu")
    return StubWhisperModel


def test_resolve_compute_type() -> None:
    assert engine.resolve_compute_type("cuda", None) == "float16"
    assert engine.resolve_compute_type("cpu", None) == "int8"
    assert engine.resolve_compute_type("cpu", "float32") == "float32"


def test_resolve_device_explicit() -> None:
    assert engine.resolve_device("cpu") == "cpu"
    assert engine.resolve_device("cuda") == "cuda"


def test_resolve_device_auto_without_gpu() -> None:
    assert engine.resolve_device("auto") in {"cpu", "cuda"}


def test_to_segment_maps_and_rounds_words() -> None:
    word = SimpleNamespace(start=0.11111, end=0.55555, word=" Hi", probability=0.912345)
    segment = engine.to_segment(4, raw_segment(0.1, 0.6, " Hi ", [word]))
    assert segment.id == 4
    assert segment.text == "Hi"
    assert segment.words is not None
    assert segment.words[0].start == 0.111
    assert segment.words[0].probability == 0.9123


def test_transcriber_loads_and_transcribes(stub_model: type[StubWhisperModel]) -> None:
    settings = Settings(_env_file=None, cpu_threads=3, beam_size=2)  # type: ignore[call-arg]
    transcriber = engine.FasterWhisperTranscriber(settings)
    model = stub_model.instances[0]
    assert model.name == settings.model_name
    assert model.kwargs == {"device": "cpu", "compute_type": "int8", "cpu_threads": 3}
    assert transcriber.model_name == settings.model_name

    info, segments = transcriber.transcribe(
        np.zeros(16000, dtype=np.float32), language="en", word_timestamps=True
    )
    assert (info.language, info.language_probability, info.duration) == ("en", 0.9877, 2.0)
    assert [(s.id, s.text, s.end) for s in segments] == [(0, "Hello", 1.235), (1, "world", 2.0)]
    call = model.calls[0]
    assert call["beam_size"] == 2
    assert call["vad_filter"] is True
    assert call["word_timestamps"] is True


def test_warmup_runs_inference(stub_model: type[StubWhisperModel]) -> None:
    transcriber = engine.FasterWhisperTranscriber(Settings(_env_file=None))  # type: ignore[call-arg]
    transcriber.warmup()
    assert len(stub_model.instances[0].calls) == 2
