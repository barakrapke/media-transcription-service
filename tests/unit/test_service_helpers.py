import pytest

from transcriber.api.service import clean_filename, is_valid_job_id, normalize_language
from transcriber.core.errors import UnsupportedLanguageError


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("auto", None), (" AUTO ", None), ("EN", "en"), ("he", "he")],
)
def test_normalize_language(value: str | None, expected: str | None) -> None:
    assert normalize_language(value) == expected


def test_normalize_language_rejects_unknown() -> None:
    with pytest.raises(UnsupportedLanguageError):
        normalize_language("klingon")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("meeting.mp4", "meeting.mp4"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\talk.wav", "talk.wav"),
        ("bad\x00name\n.mp3", "badname.mp3"),
        ("ראיון.mp3", "ראיון.mp3"),
        (None, "upload"),
        ("", "upload"),
    ],
)
def test_clean_filename(raw: str | None, expected: str) -> None:
    assert clean_filename(raw) == expected


def test_clean_filename_truncates() -> None:
    assert len(clean_filename("a" * 1000 + ".mp3")) == 255


def test_is_valid_job_id() -> None:
    assert is_valid_job_id("0123456789abcdef0123456789abcdef")
    assert not is_valid_job_id("0123456789ABCDEF0123456789ABCDEF")
    assert not is_valid_job_id("../x")
    assert not is_valid_job_id("")
