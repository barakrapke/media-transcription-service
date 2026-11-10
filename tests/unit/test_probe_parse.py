from transcriber.media.probe import parse_probe_output


def test_audio_file() -> None:
    info = parse_probe_output(
        {"format": {"format_name": "mp3", "duration": "12.5"}, "streams": [{"codec_type": "audio"}]}
    )
    assert info.format_name == "mp3"
    assert info.duration == 12.5
    assert info.has_audio and not info.has_video


def test_video_without_audio() -> None:
    info = parse_probe_output({"format": {"duration": "3"}, "streams": [{"codec_type": "video"}]})
    assert info.has_video and not info.has_audio


def test_duration_falls_back_to_streams() -> None:
    info = parse_probe_output(
        {
            "format": {"format_name": "webm"},
            "streams": [
                {"codec_type": "audio", "duration": "4.0"},
                {"codec_type": "video", "duration": "5.5"},
            ],
        }
    )
    assert info.duration == 5.5


def test_unknown_duration() -> None:
    info = parse_probe_output({"format": {"duration": "N/A"}, "streams": [{"codec_type": "audio"}]})
    assert info.duration is None


def test_empty_output() -> None:
    info = parse_probe_output({})
    assert info.format_name == "unknown"
    assert info.duration is None
    assert not info.has_audio
