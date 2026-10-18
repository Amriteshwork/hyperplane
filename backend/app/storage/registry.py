"""`STORAGE_BACKENDS[name]` → factory. Adding a backend = one new entry."""

from collections.abc import Callable

from app.core.config import Settings
from app.storage.base import StorageBackend
from app.storage.local import LocalStorage
from app.storage.s3 import S3Storage

StorageFactory = Callable[[Settings], StorageBackend]


def _local(settings: Settings) -> StorageBackend:
    return LocalStorage(settings.local_storage_root)


def _s3(settings: Settings) -> StorageBackend:
    secret = settings.s3_secret_access_key
    return S3Storage(
        settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        access_key_id=settings.s3_access_key_id,
        secret_access_key=secret.get_secret_value() if secret else None,
    )


STORAGE_BACKENDS: dict[str, StorageFactory] = {"local": _local, "s3": _s3}


def build_storage(settings: Settings) -> StorageBackend:
    """Instantiate the backend named by `settings.storage_backend`."""
    try:
        factory = STORAGE_BACKENDS[settings.storage_backend]
    except KeyError:
        known = ", ".join(sorted(STORAGE_BACKENDS))
        raise ValueError(
            f"Unknown STORAGE_BACKEND {settings.storage_backend!r} (known: {known})"
        ) from None
    return factory(settings)
