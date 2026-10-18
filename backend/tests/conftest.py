"""Shared fixtures."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import deps
from app.core.config import Environment, Settings
from app.jobs.names import PROFILE_DATASET
from app.main import create_app
from app.services.profiling_service import ProfilingService
from tests.fakes.clock import FixedClock
from tests.fakes.jobs import FakeJobQueue
from tests.fakes.repositories import InMemoryDatasetRepository, InMemorySessionRepository
from tests.fakes.storage import InMemoryStorage

# Nothing listens on port 1, so real probes fail fast instead of hanging.
UNREACHABLE_DB = "postgresql+asyncpg://test:test@127.0.0.1:1/test"
UNREACHABLE_REDIS = "redis://127.0.0.1:1/0"
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings_overrides() -> dict[str, object]:
    """Override per test with `@pytest.mark.parametrize("settings_overrides", [{...}])`."""
    return {}


@pytest.fixture
def settings(tmp_path: Path, settings_overrides: dict[str, object]) -> Settings:
    """Isolated settings: no .env, local storage in a temp dir, unreachable db/redis."""
    base = Settings(
        _env_file=None,
        app_env=Environment.TEST,
        database_url=UNREACHABLE_DB,
        redis_url=UNREACHABLE_REDIS,
        storage_backend="local",
        local_storage_root=tmp_path / "storage",
        datasets_dir=DATASETS_DIR,
        health_timeout_seconds=2.0,
    )
    return base.model_copy(update=settings_overrides)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def session_repo() -> InMemorySessionRepository:
    return InMemorySessionRepository()


@pytest.fixture
def dataset_repo() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


@pytest.fixture
def job_queue(dataset_repo: InMemoryDatasetRepository, storage: InMemoryStorage) -> FakeJobQueue:
    """Records jobs; `drain()` runs profiling against the same fakes the API uses."""
    queue = FakeJobQueue()

    def profile(dataset_id: str) -> None:
        asyncio.run(ProfilingService(dataset_repo, storage).run(UUID(dataset_id)))

    queue.handlers[PROFILE_DATASET] = profile
    return queue


@pytest.fixture
def app(
    settings: Settings,
    clock: FixedClock,
    storage: InMemoryStorage,
    session_repo: InMemorySessionRepository,
    dataset_repo: InMemoryDatasetRepository,
    job_queue: FakeJobQueue,
) -> FastAPI:
    """The real app with infrastructure dependencies swapped for in-memory fakes."""
    app = create_app(settings)
    app.dependency_overrides.update(
        {
            deps.get_clock: lambda: clock,
            deps.get_storage: lambda: storage,
            deps.get_blob_reader: lambda: storage,
            deps.get_session_repository: lambda: session_repo,
            deps.get_dataset_repository: lambda: dataset_repo,
            deps.get_job_queue: lambda: job_queue,
        }
    )
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
