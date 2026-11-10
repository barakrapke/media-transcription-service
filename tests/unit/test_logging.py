import json
import logging
import sys

import pytest

from transcriber.core.logging import (
    JsonFormatter,
    TextFormatter,
    _ContextFilter,
    configure_logging,
    job_id_var,
    request_id_var,
)


def _record(message: str = "hello") -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
    _ContextFilter().filter(record)
    return record


def test_json_formatter_includes_context() -> None:
    token = request_id_var.set("req-1")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "req-1"
    assert "job_id" not in payload  # None values are omitted


def test_json_formatter_includes_exceptions() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "x", None, sys.exc_info())
    _ContextFilter().filter(record)
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(record))["exc_info"]


def test_text_formatter_includes_context() -> None:
    token = job_id_var.set("job-9")
    try:
        line = TextFormatter().format(_record())
    finally:
        job_id_var.reset(token)
    assert "[job_id=job-9]" in line
    assert line.endswith("hello")


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_configure_logging(fmt: str) -> None:
    configure_logging("DEBUG", fmt)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert logging.getLogger("uvicorn.access").propagate
