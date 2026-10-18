"""Health probe for any `BlobReader`."""

import anyio.to_thread

from app.storage.base import BlobReader

# Looking up a key that never exists still proves connectivity and credentials.
_PROBE_KEY = "health/probe"


class StorageProbe:
    name = "storage"

    def __init__(self, reader: BlobReader) -> None:
        self._reader = reader

    async def check(self) -> None:
        await anyio.to_thread.run_sync(self._reader.exists, _PROBE_KEY)
