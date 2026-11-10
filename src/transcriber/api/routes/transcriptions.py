from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from transcriber.api.dependencies import get_service
from transcriber.api.schemas import ErrorResponse, JobResponse
from transcriber.api.service import TranscriptionService

router = APIRouter(prefix="/v1/transcriptions", tags=["transcriptions"])

Service = Annotated[TranscriptionService, Depends(get_service)]

_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Job not found"},
}


def _content_disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode() or "transcript"
    ascii_name = ascii_name.replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Upload a media file for transcription",
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Invalid media, no audio, too long, ..."},
        503: {"model": ErrorResponse, "description": "Queue full or dependency unavailable"},
    },
)
async def create_transcription(
    response: Response,
    service: Service,
    file: Annotated[UploadFile, File(description="Audio or video file.")],
    language: Annotated[
        str | None,
        Form(description="ISO 639-1 language code (e.g. 'en', 'he'). Omit or 'auto' to detect."),
    ] = None,
    word_timestamps: Annotated[bool, Form(description="Include word-level timestamps.")] = False,
) -> JobResponse:
    job = await service.create(file, language, word_timestamps)
    response.headers["Location"] = f"{router.prefix}/{job.id}"
    return job


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get a job, or export its transcript",
    description=(
        "Returns the job resource as JSON. With `format=txt|srt|vtt` the completed transcript "
        "is returned as a file download (409 if the job has not completed)."
    ),
    responses={
        **_ERRORS,
        200: {
            "content": {
                "text/plain": {},
                "application/x-subrip": {},
                "text/vtt": {},
            }
        },
        409: {"model": ErrorResponse, "description": "Export requested before completion"},
    },
)
async def get_transcription(
    job_id: str,
    service: Service,
    export_format: Annotated[Literal["json", "txt", "srt", "vtt"], Query(alias="format")] = "json",
) -> Any:
    if export_format == "json":
        return await service.get(job_id)
    content, media_type, filename = await service.export(job_id, export_format)
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@router.get(
    "/{job_id}/events",
    summary="Stream job progress (Server-Sent Events)",
    description=(
        "Events: `status`, `progress`, `segment`, `completed`, `failed`. The current state and "
        "all segments produced so far are replayed on connect. The stream ends when the job "
        "reaches a terminal state."
    ),
    response_class=StreamingResponse,
    responses={**_ERRORS, 200: {"content": {"text/event-stream": {}}}},
)
async def stream_transcription_events(
    job_id: str, request: Request, service: Service
) -> StreamingResponse:
    events = await service.open_event_stream(job_id, request.is_disconnected)
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete(
    "/{job_id}",
    summary="Cancel an active job, or delete a finished one",
    description=(
        "Queued or processing jobs are cancelled (202, returns the job). Finished jobs and "
        "their data are deleted (204)."
    ),
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    responses={**_ERRORS, 204: {"description": "Job deleted"}},
)
async def delete_transcription(job_id: str, service: Service) -> Response:
    job = await service.delete(job_id)
    if job is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return JSONResponse(job.model_dump(mode="json"), status_code=status.HTTP_202_ACCEPTED)
