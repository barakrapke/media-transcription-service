from __future__ import annotations

from fastapi import Request

from transcriber.api.service import TranscriptionService


def get_service(request: Request) -> TranscriptionService:
    state = request.app.state
    return TranscriptionService(state.store, state.storage, state.settings)
