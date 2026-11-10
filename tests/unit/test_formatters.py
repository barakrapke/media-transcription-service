from transcriber.formatters import EXPORT_FORMATS, _timestamp, to_srt, to_txt, to_vtt
from transcriber.jobs.models import Segment

SEGMENTS = [
    Segment(id=0, start=0.0, end=1.5, text="Hello there."),
    Segment(id=1, start=3661.2345, end=3662.0, text="Second line."),
]


def test_timestamp_formats_hours_minutes_seconds_millis() -> None:
    assert _timestamp(0, ",") == "00:00:00,000"
    assert _timestamp(3661.2345, ",") == "01:01:01,234"
    assert _timestamp(59.9996, ".") == "00:01:00.000"  # rounds up across the boundary


def test_timestamp_clamps_negative_values() -> None:
    assert _timestamp(-0.5, ",") == "00:00:00,000"


def test_srt() -> None:
    assert to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n"
        "2\n01:01:01,234 --> 01:01:02,000\nSecond line.\n"
    )


def test_vtt() -> None:
    assert to_vtt(SEGMENTS) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\nHello there.\n\n"
        "01:01:01.234 --> 01:01:02.000\nSecond line.\n"
    )


def test_txt_one_segment_per_line() -> None:
    assert to_txt(SEGMENTS) == "Hello there.\nSecond line.\n"


def test_empty_transcript() -> None:
    assert to_srt([]) == ""
    assert to_vtt([]) == "WEBVTT\n\n"
    assert to_txt([]) == ""


def test_export_registry() -> None:
    assert set(EXPORT_FORMATS) == {"txt", "srt", "vtt"}
    for render, media_type, extension in EXPORT_FORMATS.values():
        assert callable(render)
        assert "charset=utf-8" in media_type
        assert extension
