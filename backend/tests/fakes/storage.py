"""Dict-backed `StorageBackend`."""

from app.storage.base import BlobNotFoundError, validate_key


class InMemoryStorage:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def read(self, key: str) -> bytes:
        try:
            return self.blobs[validate_key(key)]
        except KeyError:
            raise BlobNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        return validate_key(key) in self.blobs

    def write(self, key: str, data: bytes) -> None:
        self.blobs[validate_key(key)] = bytes(data)

    def delete(self, key: str) -> None:
        self.blobs.pop(validate_key(key), None)
