"""Worker composition root. Job functions call `container()` to get their dependencies."""

from dataclasses import dataclass
from functools import lru_cache

from redis import Redis

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.jobs.base import EventPublisher
from app.jobs.redis_events import RedisEventPublisher
from app.ml.registry import AlgorithmRegistry
from app.storage.base import StorageBackend
from app.storage.registry import build_storage


@dataclass(frozen=True, slots=True)
class WorkerContainer:
    settings: Settings
    storage: StorageBackend
    events: EventPublisher
    registry: AlgorithmRegistry


def build_worker_container(settings: Settings) -> WorkerContainer:
    """Wire the worker's concrete implementations."""
    return WorkerContainer(
        settings=settings,
        storage=build_storage(settings),
        events=RedisEventPublisher(Redis.from_url(settings.redis_url)),
        registry=AlgorithmRegistry.discover(),
    )


@lru_cache(maxsize=1)
def container() -> WorkerContainer:
    """Built once per worker process, on the first job."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return build_worker_container(settings)
