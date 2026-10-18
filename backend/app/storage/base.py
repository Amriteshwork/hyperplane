"""Blob storage protocols. Keys are POSIX-style relative paths, e.g. `datasets/<id>/raw.csv`."""

import re
from typing import Protocol, runtime_checkable

_SEGMENT = re.compile(r"^[A-Za-z0-9._=-]+$")


class BlobNotFoundError(KeyError):
    """No blob is stored under the requested key."""


class InvalidBlobKeyError(ValueError):
    """The key is empty, absolute, or escapes the storage root."""


def validate_key(key: str) -> str:
    """Return `key` if it is a safe relative key, else raise `InvalidBlobKeyError`."""
    segments = key.split("/")
    if not key or any(s in ("", ".", "..") or not _SEGMENT.match(s) for s in segments):
        raise InvalidBlobKeyError(key)
    return key


@runtime_checkable
class BlobReader(Protocol):
    def read(self, key: str) -> bytes:
        """Return the blob's bytes; raise `BlobNotFoundError` if missing."""
        ...

    def exists(self, key: str) -> bool:
        """Whether a blob is stored under `key`."""
        ...


@runtime_checkable
class BlobWriter(Protocol):
    def write(self, key: str, data: bytes) -> None:
        """Create or replace the blob at `key`."""
        ...

    def delete(self, key: str) -> None:
        """Remove the blob; deleting a missing key is a no-op."""
        ...


class StorageBackend(BlobReader, BlobWriter, Protocol):
    """Read + write. Callers that only download should depend on `BlobReader`."""
