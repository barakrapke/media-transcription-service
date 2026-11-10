"""Public API response models (the wire format; independent of the internal Job model)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from transcriber.jobs.models import Job, JobErrorInfo, JobStatus, Segment


class TranscriptionResult(BaseModel):
    text: str = Field(description="Full transcript.")
    language: str = Field(description="Detected (or requested) language code.")
    language_probability: float
    duration: float = Field(description="Audio duration in seconds.")
    model: str
    processing_time: float = Field(description="Wall-clock processing time in seconds.")
    segments: list[Segment]


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress: float = Field(description="0..1, estimated from the position of the last segment.")
    filename: str
    language: str | None = Field(description="Requested language; null means auto-detect.")
    word_timestamps: bool
    attempts: int = Field(
        description="Processing attempts so far (a job is retried if its worker dies)."
    )
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: JobErrorInfo | None
    result: TranscriptionResult | None = Field(description="Present once status is completed.")

    @classmethod
    def from_job(cls, job: Job, segments: list[Segment]) -> JobResponse:
        result = None
        if job.status is JobStatus.COMPLETED and job.summary is not None:
            result = TranscriptionResult(**job.summary.model_dump(), segments=segments)
        return cls(
            id=job.id,
            status=job.status,
            progress=job.progress,
            filename=job.filename,
            language=job.language,
            word_timestamps=job.word_timestamps,
            attempts=job.attempts,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            result=result,
        )


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    redis: bool
    workers: int
    queue_length: int | None = None
