"""Worker composition root. Job functions get their dependencies from `container()`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache

from redis import Redis
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.repositories.sql_datasets import SqlDatasetRepository
from app.jobs.base import EventPublisher
from app.jobs.redis_events import RedisEventPublisher
from app.ml.registry import AlgorithmRegistry
from app.services.profiling_service import ProfilingService
from app.storage.base import StorageBackend
from app.storage.registry import build_storage


@dataclass(frozen=True, slots=True)
class WorkerContainer:
    settings: Settings
    storage: StorageBackend
    events: EventPublisher
    registry: AlgorithmRegistry
    session_factory: async_sessionmaker[AsyncSession]

    @asynccontextmanager
    async def profiling_service(self) -> AsyncIterator[ProfilingService]:
        async with self.session_factory() as db:
            yield ProfilingService(SqlDatasetRepository(db), self.storage)


def build_worker_container(settings: Settings) -> WorkerContainer:
    """Wire the worker's concrete implementations."""
    # Each job runs its own event loop (asyncio.run), so pooled connections can't be reused.
    engine = create_async_engine(settings.database_url, poolclass=pool.NullPool)
    return WorkerContainer(
        settings=settings,
        storage=build_storage(settings),
        events=RedisEventPublisher(Redis.from_url(settings.redis_url)),
        registry=AlgorithmRegistry.discover(),
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
    )


@lru_cache(maxsize=1)
def container() -> WorkerContainer:
    """Built once per worker process, on the first job."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return build_worker_container(settings)
