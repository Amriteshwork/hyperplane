"""API composition root: the only place concrete infrastructure is instantiated.

`build_container` runs once in the app lifespan; FastAPI dependencies below hand out
protocol-typed pieces of it. Tests override these dependencies with fakes.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.probe import DatabaseProbe
from app.db.session import create_engine, create_session_factory
from app.jobs.base import JobQueue
from app.jobs.probe import RedisProbe
from app.jobs.rq_queue import RqJobQueue
from app.ml.registry import AlgorithmRegistry
from app.services.health_service import HealthService
from app.storage.base import BlobReader, StorageBackend
from app.storage.probe import StorageProbe
from app.storage.registry import build_storage

_CONNECT_TIMEOUT = 2.0
_SOCKET_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: AsyncRedis
    queue_redis: Redis
    storage: StorageBackend
    registry: AlgorithmRegistry

    async def aclose(self) -> None:
        await self.engine.dispose()
        await self.redis.aclose()
        self.queue_redis.close()


def build_container(settings: Settings) -> Container:
    """Wire every concrete implementation. Nothing here does I/O."""
    engine = create_engine(settings.database_url)
    return Container(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        redis=AsyncRedis.from_url(
            settings.redis_url,
            socket_connect_timeout=_CONNECT_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
        ),
        queue_redis=Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=_CONNECT_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
        ),
        storage=build_storage(settings),
        registry=AlgorithmRegistry.discover(),
    )


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    return container.settings


def get_storage(container: ContainerDep) -> StorageBackend:
    return container.storage


def get_blob_reader(container: ContainerDep) -> BlobReader:
    return container.storage


def get_registry(container: ContainerDep) -> AlgorithmRegistry:
    return container.registry


def get_job_queue(container: ContainerDep) -> JobQueue:
    settings = container.settings
    return RqJobQueue(
        container.queue_redis,
        name=settings.job_queue_name,
        timeout_seconds=settings.job_timeout_seconds,
    )


async def get_db_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    async with container.session_factory() as session:
        yield session


def get_health_service(container: ContainerDep) -> HealthService:
    return HealthService(
        [
            DatabaseProbe(container.engine),
            RedisProbe(container.redis),
            StorageProbe(container.storage),
        ],
        timeout_seconds=container.settings.health_timeout_seconds,
    )
