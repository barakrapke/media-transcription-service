from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from transcriber.api.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    summary="Readiness probe: Redis reachable and at least one live worker",
    responses={503: {"model": ReadinessResponse}},
)
async def readyz(request: Request) -> JSONResponse:
    store = request.app.state.store
    try:
        await store.redis.ping()
        workers = await store.count_live_workers()
        queue_length = await store.queue_length()
        redis_ok = True
    except RedisError:
        redis_ok, workers, queue_length = False, 0, None
    ready = redis_ok and workers > 0
    body = ReadinessResponse(
        status="ready" if ready else "not_ready",
        redis=redis_ok,
        workers=workers,
        queue_length=queue_length,
    )
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)
