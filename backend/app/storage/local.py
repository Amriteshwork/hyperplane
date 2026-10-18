"""Local-disk storage for development."""

import os
import tempfile
from pathlib import Path

from app.storage.base import BlobNotFoundError, validate_key


class LocalStorage:
    """Blobs are files under `root`; writes are atomic (temp file + rename)."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, key: str) -> Path:
        return self._root / validate_key(key)

    def read(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise BlobNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
