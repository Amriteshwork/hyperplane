"""API composition root: the only place concrete infrastructure is instantiated.

`build_container` runs once in the app lifespan; FastAPI dependencies below hand out
protocol-typed pieces of it. Tests override these dependencies with fakes.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, Request, Response
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.content.catalog import ContentCatalog
from app.core.clock import Clock, SystemClock
from app.core.config import Settings
from app.db.probe import DatabaseProbe
from app.db.repositories.base import DatasetRepository, RunRepository, SessionRepository
from app.db.repositories.sql_datasets import SqlDatasetRepository
from app.db.repositories.sql_runs import SqlRunRepository
from app.db.repositories.sql_sessions import SqlSessionRepository
from app.db.session import create_engine, create_session_factory
from app.ingest.readers import TableLimits
from app.jobs.base import EventSubscriber, JobQueue
from app.jobs.probe import RedisProbe
from app.jobs.redis_events import RedisEventSubscriber
from app.jobs.rq_queue import RqJobQueue
from app.ml.registry import AlgorithmRegistry
from app.samples.catalog import SampleCatalog
from app.services.content_service import ContentService
from app.services.dataset_service import DatasetService, UploadPolicy
from app.services.experiment_service import ExperimentService
from app.services.health_service import HealthService
from app.services.session_service import SessionService
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
    samples: SampleCatalog
    content: ContentCatalog
    clock: Clock

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
        samples=SampleCatalog.load(settings.datasets_dir),
        content=ContentCatalog.load(settings.content_dir),
        clock=SystemClock(),
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


def get_event_subscriber(container: ContainerDep) -> EventSubscriber:
    # A stream outlives its run by at most the job timeout (a stuck worker can't hold it open).
    return RedisEventSubscriber(
        container.redis, max_seconds=container.settings.job_timeout_seconds + 60
    )


def get_health_service(container: ContainerDep) -> HealthService:
    return HealthService(
        [
            DatabaseProbe(container.engine),
            RedisProbe(container.redis),
            StorageProbe(container.storage),
        ],
        timeout_seconds=container.settings.health_timeout_seconds,
    )


def get_clock(container: ContainerDep) -> Clock:
    return container.clock


def get_sample_catalog(container: ContainerDep) -> SampleCatalog:
    return container.samples


def get_content_service(container: ContainerDep) -> ContentService:
    return ContentService(container.content, container.registry)


# Function scope: the session closes when the handler returns, not when the response ends,
# so streaming responses (SSE) don't hold a database connection.
DbSession = Annotated[AsyncSession, Depends(get_db_session, scope="function")]


def get_session_repository(db: DbSession) -> SessionRepository:
    return SqlSessionRepository(db)


def get_dataset_repository(db: DbSession) -> DatasetRepository:
    return SqlDatasetRepository(db)


def get_run_repository(db: DbSession) -> RunRepository:
    return SqlRunRepository(db)


def get_session_service(
    sessions: Annotated[SessionRepository, Depends(get_session_repository)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> SessionService:
    return SessionService(sessions, clock)


def get_dataset_service(
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    storage: Annotated[StorageBackend, Depends(get_storage)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    clock: Annotated[Clock, Depends(get_clock)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatasetService:
    policy = UploadPolicy(
        max_bytes=settings.max_upload_mb * 2**20,
        limits=TableLimits(max_rows=settings.max_rows, max_columns=settings.max_columns),
        retention=timedelta(days=settings.retention_days),
    )
    return DatasetService(datasets, storage, queue, clock, policy)


def get_experiment_service(
    datasets: Annotated[DatasetService, Depends(get_dataset_service)],
    runs: Annotated[RunRepository, Depends(get_run_repository)],
    registry: Annotated[AlgorithmRegistry, Depends(get_registry)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    events: Annotated[EventSubscriber, Depends(get_event_subscriber)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> ExperimentService:
    return ExperimentService(datasets, runs, registry, queue, events, clock)


SESSION_HEADER = "X-Session-ID"


async def get_session_id(
    request: Request,
    response: Response,
    service: Annotated[SessionService, Depends(get_session_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UUID:
    """Anonymous session from the X-Session-ID header or cookie; starts one if absent."""
    raw = request.headers.get(SESSION_HEADER) or request.cookies.get(settings.session_cookie_name)
    session_id = await service.resolve(raw)
    response.set_cookie(
        settings.session_cookie_name,
        str(session_id),
        max_age=settings.retention_days * 86_400,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
    )
    response.headers[SESSION_HEADER] = str(session_id)
    return session_id


SessionId = Annotated[UUID, Depends(get_session_id)]
