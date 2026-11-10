from transcriber.api.sse import format_event


def test_format_event_with_id() -> None:
    assert format_event("segment", {"text": "héllo"}, event_id="3") == (
        'event: segment\nid: 3\ndata: {"text":"héllo"}\n\n'
    )


def test_format_event_without_id() -> None:
    assert format_event("status", {"status": "queued"}) == (
        'event: status\ndata: {"status":"queued"}\n\n'
    )
