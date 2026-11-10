"""Upload storage on the volume shared by the API and the workers.

Files are stored under the job id (a UUID) - never under the user-supplied filename - which
rules out path traversal and name collisions. The original filename is kept as job metadata.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from transcriber.core.errors import FileTooLargeError

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024


class UploadStorage:
    def __init__(self, root: Path):
        self._root = root

    def ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return name.isascii() and name.isalnum()  # job ids are uuid4 hex

    def path_for(self, job_id: str) -> Path:
        if not self.is_valid_name(job_id):  # refuse anything else defensively
            raise ValueError(f"invalid job id: {job_id!r}")
        return self._root / job_id

    def save(self, job_id: str, source: BinaryIO, max_bytes: int) -> int:
        """Copy ``source`` to storage in chunks, enforcing ``max_bytes``. Returns bytes written."""
        self.ensure_root()
        target = self.path_for(job_id)
        written = 0
        try:
            with target.open("wb") as out:
                while chunk := source.read(_CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        limit_mb = max_bytes // (1024 * 1024)
                        raise FileTooLargeError(f"The file is larger than the {limit_mb} MB limit.")
                    out.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return written

    def delete(self, job_id: str) -> None:
        self.path_for(job_id).unlink(missing_ok=True)

    def list_older_than(self, age_seconds: float) -> list[str]:
        """Job ids of stored files whose modification time is older than ``age_seconds``.
        Files that can't be uploads (e.g. ``.DS_Store``) are ignored."""
        if not self._root.exists():
            return []
        cutoff = time.time() - age_seconds
        return [
            path.name
            for path in self._root.iterdir()
            if self.is_valid_name(path.name) and path.is_file() and path.stat().st_mtime < cutoff
        ]

    def free_bytes(self) -> int:
        self.ensure_root()
        return shutil.disk_usage(self._root).free
