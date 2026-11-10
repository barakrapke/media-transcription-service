"""Application settings, loaded from environment variables (and an optional ``.env`` file).

Settings are validated once at startup so that a misconfiguration fails fast with a clear
message instead of surfacing later as a confusing runtime error.
"""

from __future__ import annotations

import socket
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # --- Infrastructure -------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("./data")

    # --- API ------------------------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - the API is meant to listen on all interfaces in a container
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_upload_mb: int = Field(default=200, gt=0)
    max_duration_seconds: int = Field(default=3600, gt=0)
    max_queue_size: int = Field(default=20, gt=0)
    probe_timeout_seconds: float = Field(default=30, gt=0)
    sse_poll_interval_seconds: float = Field(default=0.5, gt=0)
    sse_keepalive_seconds: float = Field(default=15, gt=0)

    # --- Jobs -----------------------------------------------------------------------------
    result_ttl_seconds: int = Field(default=86_400, gt=0)
    job_timeout_seconds: int = Field(default=3600, gt=0)
    decode_timeout_seconds: int = Field(default=600, gt=0)
    max_attempts: int = Field(default=2, ge=1)

    # --- Worker ---------------------------------------------------------------------------
    worker_id: str = Field(default_factory=socket.gethostname)
    heartbeat_interval_seconds: float = Field(default=10, gt=0)
    heartbeat_ttl_seconds: float = Field(default=30, gt=0)
    sweep_interval_seconds: float = Field(default=30, gt=0)
    claim_timeout_seconds: int = Field(default=1, ge=1)

    # --- Model ----------------------------------------------------------------------------
    model_name: str = "Systran/faster-whisper-small"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str | None = None  # None = float16 on CUDA, int8 on CPU
    cpu_threads: int = Field(default=0, ge=0)  # 0 = let CTranslate2 decide
    beam_size: int = Field(default=5, ge=1)
    model_warmup: bool = True

    # --- Logging --------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"

    @model_validator(mode="after")
    def _check_heartbeat(self) -> Settings:
        if self.heartbeat_ttl_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "HEARTBEAT_TTL_SECONDS must be greater than HEARTBEAT_INTERVAL_SECONDS"
            )
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
