"""Error model shared by the API and the worker.

* ``AppError`` subclasses are raised by the API layer and rendered as a uniform JSON error body.
* ``JobFailure`` subclasses are raised inside the worker and recorded on the job (``job.error``).

Every error carries a stable machine-readable ``code`` so clients never have to parse messages.
"""

from __future__ import annotations

from enum import StrEnum
from http import HTTPStatus


class ErrorCode(StrEnum):
    # API errors
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_MEDIA = "INVALID_MEDIA"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    DURATION_TOO_LONG = "DURATION_TOO_LONG"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_COMPLETED = "JOB_NOT_COMPLETED"
    QUEUE_FULL = "QUEUE_FULL"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # Job (worker) errors
    DECODE_ERROR = "DECODE_ERROR"
    DECODE_TIMEOUT = "DECODE_TIMEOUT"
    TRANSCRIPTION_ERROR = "TRANSCRIPTION_ERROR"
    JOB_TIMEOUT = "JOB_TIMEOUT"
    WORKER_LOST = "WORKER_LOST"


class AppError(Exception):
    """Base class for errors that are returned to API clients."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, headers: dict[str, str] | None = None):
        self.message = message or self.default_message
        self.headers = headers
        super().__init__(self.message)


class FileTooLargeError(AppError):
    code = ErrorCode.FILE_TOO_LARGE
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    default_message = "The uploaded file exceeds the maximum allowed size."


class InvalidMediaError(AppError):
    code = ErrorCode.INVALID_MEDIA
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "The file could not be read as an audio or video file."


class NoAudioStreamError(AppError):
    code = ErrorCode.NO_AUDIO_STREAM
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "The file has no audio track."


class DurationTooLongError(AppError):
    code = ErrorCode.DURATION_TOO_LONG
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "The media is longer than the maximum allowed duration."


class UnsupportedLanguageError(AppError):
    code = ErrorCode.UNSUPPORTED_LANGUAGE
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "The requested language is not supported."


class JobNotFoundError(AppError):
    code = ErrorCode.JOB_NOT_FOUND
    status_code = HTTPStatus.NOT_FOUND
    default_message = "Transcription job not found."


class JobNotCompletedError(AppError):
    code = ErrorCode.JOB_NOT_COMPLETED
    status_code = HTTPStatus.CONFLICT
    default_message = "The transcription job has not completed yet."


class QueueFullError(AppError):
    code = ErrorCode.QUEUE_FULL
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    default_message = "The service is at capacity. Please retry later."


class DependencyUnavailableError(AppError):
    code = ErrorCode.DEPENDENCY_UNAVAILABLE
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    default_message = "A backing service is unavailable. Please retry later."


class JobFailure(Exception):
    """Base class for errors that terminate a job inside the worker."""

    code: ErrorCode = ErrorCode.TRANSCRIPTION_ERROR
    default_message: str = "Transcription failed."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class DecodeError(JobFailure):
    code = ErrorCode.DECODE_ERROR
    default_message = "The media could not be decoded."


class DecodeTimeoutError(JobFailure):
    code = ErrorCode.DECODE_TIMEOUT
    default_message = "Decoding the media took too long."


class TranscriptionError(JobFailure):
    code = ErrorCode.TRANSCRIPTION_ERROR
    default_message = "The transcription model failed to process the audio."


class JobTimeoutError(JobFailure):
    code = ErrorCode.JOB_TIMEOUT
    default_message = "The transcription job exceeded the maximum processing time."
