from pathlib import Path

import pytest
from pydantic import ValidationError

from transcriber.core.config import Settings


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)  # set in the Docker image
    settings = make()
    assert settings.model_name == "Systran/faster-whisper-small"
    assert settings.max_upload_bytes == 200 * 1024 * 1024
    assert settings.max_duration_seconds == 3600
    assert settings.max_queue_size == 20
    assert settings.upload_dir == Path("data/uploads")


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("DEVICE", "cpu")
    settings = make()
    assert settings.max_upload_bytes == 7 * 1024 * 1024
    assert settings.device == "cpu"


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_upload_mb": 0},
        {"device": "tpu"},
        {"log_format": "xml"},
        {"heartbeat_interval_seconds": 30, "heartbeat_ttl_seconds": 10},
    ],
)
def test_invalid_values_fail_fast(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make(**overrides)
