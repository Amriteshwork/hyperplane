"""Shared `StorageBackend` contract (ARCHITECTURE §12 L): every backend must pass it.

S3Storage runs only with HYPERPLANE_TEST_S3=1 against a live endpoint (e.g. `make up`),
using the S3_* settings from the environment / .env.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.core.config import Settings
from app.storage.base import (
    BlobNotFoundError,
    BlobReader,
    BlobWriter,
    InvalidBlobKeyError,
    StorageBackend,
)
from app.storage.local import LocalStorage
from app.storage.registry import build_storage
from tests.fakes.storage import InMemoryStorage


def _s3() -> StorageBackend:
    return build_storage(Settings(storage_backend="s3"))


BACKENDS: dict[str, Callable[[Path], StorageBackend]] = {
    "local": lambda tmp: LocalStorage(tmp / "blobs"),
    "memory": lambda _tmp: InMemoryStorage(),
    "s3": lambda _tmp: _s3(),
}


@pytest.fixture(
    params=[
        "local",
        "memory",
        pytest.param(
            "s3",
            marks=[
                pytest.mark.s3,
                pytest.mark.skipif(
                    os.environ.get("HYPERPLANE_TEST_S3") != "1", reason="HYPERPLANE_TEST_S3!=1"
                ),
            ],
        ),
    ]
)
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> StorageBackend:
    backend: StorageBackend = BACKENDS[request.param](tmp_path)
    return backend


@pytest.fixture
def prefix(storage: StorageBackend) -> Iterator[str]:
    """Unique key prefix per test, so a shared bucket stays isolated; cleaned up after."""
    value = f"contract-{uuid.uuid4().hex}"
    keys: list[str] = []
    original_write = storage.write

    def tracking_write(key: str, data: bytes) -> None:
        keys.append(key)
        original_write(key, data)

    storage.write = tracking_write  # type: ignore[method-assign]
    yield value
    for key in keys:
        storage.delete(key)


def test_implements_reader_and_writer(storage: StorageBackend) -> None:
    assert isinstance(storage, BlobReader)
    assert isinstance(storage, BlobWriter)


def test_write_then_read_roundtrip(storage: StorageBackend, prefix: str) -> None:
    data = bytes(range(256)) * 4  # binary-safe, includes NUL bytes
    storage.write(f"{prefix}/nested/dir/blob.bin", data)
    assert storage.read(f"{prefix}/nested/dir/blob.bin") == data


def test_write_overwrites(storage: StorageBackend, prefix: str) -> None:
    storage.write(f"{prefix}/a.txt", b"first")
    storage.write(f"{prefix}/a.txt", b"second")
    assert storage.read(f"{prefix}/a.txt") == b"second"


def test_empty_blob(storage: StorageBackend, prefix: str) -> None:
    storage.write(f"{prefix}/empty", b"")
    assert storage.exists(f"{prefix}/empty")
    assert storage.read(f"{prefix}/empty") == b""


def test_exists(storage: StorageBackend, prefix: str) -> None:
    assert not storage.exists(f"{prefix}/a.txt")
    storage.write(f"{prefix}/a.txt", b"x")
    assert storage.exists(f"{prefix}/a.txt")


def test_read_missing_raises(storage: StorageBackend, prefix: str) -> None:
    with pytest.raises(BlobNotFoundError):
        storage.read(f"{prefix}/missing.txt")


def test_delete(storage: StorageBackend, prefix: str) -> None:
    storage.write(f"{prefix}/a.txt", b"x")
    storage.delete(f"{prefix}/a.txt")
    assert not storage.exists(f"{prefix}/a.txt")


def test_delete_missing_is_noop(storage: StorageBackend, prefix: str) -> None:
    storage.delete(f"{prefix}/never-written.txt")


@pytest.mark.parametrize(
    "key", ["", "/abs/path", "../escape", "a/../b", "a//b", "a/./b", "trailing/", "sp ace", "ü"]
)
def test_rejects_unsafe_keys(storage: StorageBackend, key: str) -> None:
    with pytest.raises(InvalidBlobKeyError):
        storage.write(key, b"x")
    with pytest.raises(InvalidBlobKeyError):
        storage.read(key)
