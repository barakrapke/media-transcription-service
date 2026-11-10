import io
import os
import time
from pathlib import Path

import pytest

from transcriber.core.errors import FileTooLargeError
from transcriber.media.storage import UploadStorage

JOB_ID = "0123456789abcdef0123456789abcdef"


def test_save_and_delete(tmp_path: Path) -> None:
    storage = UploadStorage(tmp_path / "uploads")
    written = storage.save(JOB_ID, io.BytesIO(b"x" * 3000), max_bytes=10_000)
    assert written == 3000
    assert storage.path_for(JOB_ID).read_bytes() == b"x" * 3000
    storage.delete(JOB_ID)
    assert not storage.path_for(JOB_ID).exists()
    storage.delete(JOB_ID)  # idempotent


def test_save_enforces_limit_and_cleans_up(tmp_path: Path) -> None:
    storage = UploadStorage(tmp_path)
    with pytest.raises(FileTooLargeError):
        storage.save(JOB_ID, io.BytesIO(b"x" * 3 * 1024 * 1024), max_bytes=1024 * 1024)
    assert not storage.path_for(JOB_ID).exists()


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", "abc.def"])
def test_path_for_rejects_unsafe_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="invalid job id"):
        UploadStorage(tmp_path).path_for(bad)


def test_list_older_than(tmp_path: Path) -> None:
    storage = UploadStorage(tmp_path)
    assert UploadStorage(tmp_path / "missing").list_older_than(0) == []
    storage.save("old", io.BytesIO(b"1"), max_bytes=10)
    storage.save("new", io.BytesIO(b"1"), max_bytes=10)
    past = time.time() - 3600
    os.utime(storage.path_for("old"), (past, past))
    assert storage.list_older_than(60) == ["old"]
    assert storage.free_bytes() > 0
