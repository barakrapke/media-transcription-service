"""Job domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None


class TranscriptionSummary(BaseModel):
    """Result metadata stored on the job hash. Segments are stored separately (see JobStore)."""

    text: str
    language: str
    language_probability: float
    duration: float
    model: str
    processing_time: float


class JobErrorInfo(BaseModel):
    code: str
    message: str


class Job(BaseModel):
    id: str
    status: JobStatus
    filename: str
    language: str | None = None
    word_timestamps: bool = False
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = 0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    claimed_at: float | None = None  # epoch seconds; used by the recovery sweeper
    error: JobErrorInfo | None = None
    summary: TranscriptionSummary | None = None
