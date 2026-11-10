"""Pure ASGI middleware (no BaseHTTPMiddleware, so streaming responses are unaffected)."""

from __future__ import annotations

import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from transcriber.api.errors import error_response
from transcriber.core.errors import FileTooLargeError
from transcriber.core.logging import request_id_var

_REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestIdMiddleware:
    """Accepts a sane client ``X-Request-ID`` or generates one; exposes it to logs and echoes it."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get(_REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(_REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_id)


class MaxBodySizeMiddleware:
    """Rejects oversized uploads with 413 while they stream in, before they are fully buffered.

    A declared ``Content-Length`` above the limit is rejected immediately. Otherwise bytes are
    counted as they arrive; once the limit is crossed the app sees a client disconnect, its
    response is suppressed and a 413 is sent instead.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int, paths: set[str]):
        self.app = app
        self.max_bytes = max_bytes
        self.paths = paths

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        err = FileTooLargeError()
        response = error_response(err.status_code, err.code, err.message)
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not response_started:
            await self._reject(scope, receive, send)
