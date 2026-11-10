"""Exception handlers: every error leaves the API in the same JSON shape (see ErrorResponse)."""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException as StarletteHTTPException

from transcriber.api.schemas import ErrorBody, ErrorDetail, ErrorResponse
from transcriber.core.errors import AppError, DependencyUnavailableError, ErrorCode
from transcriber.core.logging import request_id_var

logger = logging.getLogger(__name__)

_HTTP_CODES = {
    HTTPStatus.BAD_REQUEST: ErrorCode.BAD_REQUEST,
    HTTPStatus.NOT_FOUND: ErrorCode.NOT_FOUND,
    HTTPStatus.METHOD_NOT_ALLOWED: ErrorCode.METHOD_NOT_ALLOWED,
}


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=code, message=message, request_id=request_id_var.get(), details=details
        )
    )
    return JSONResponse(
        body.model_dump(exclude_none=True), status_code=status_code, headers=headers
    )


async def _app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return error_response(exc.status_code, exc.code, exc.message, headers=exc.headers)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in err.get("loc", ()) if part != "body"),
            message=err.get("msg", "invalid value"),
        )
        for err in exc.errors()
    ]
    return error_response(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        ErrorCode.VALIDATION_ERROR,
        "The request is invalid.",
        details=details,
    )


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(HTTPStatus(exc.status_code), ErrorCode.BAD_REQUEST)
    headers = dict(exc.headers) if exc.headers else None
    return error_response(exc.status_code, code, str(exc.detail), headers=headers)


async def _redis_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Redis unavailable: %s", exc)
    err = DependencyUnavailableError()
    return error_response(err.status_code, err.code, err.message, headers={"Retry-After": "5"})


async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=exc)
    return error_response(
        HTTPStatus.INTERNAL_SERVER_ERROR, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred."
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RedisError, _redis_error)
    app.add_exception_handler(Exception, _unhandled_error)
