"""Shared fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Environment, Settings
from app.main import create_app

# Nothing listens on port 1, so real probes fail fast instead of hanging.
UNREACHABLE_DB = "postgresql+asyncpg://test:test@127.0.0.1:1/test"
UNREACHABLE_REDIS = "redis://127.0.0.1:1/0"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: no .env, local storage in a temp dir, unreachable db/redis."""
    return Settings(
        _env_file=None,
        app_env=Environment.TEST,
        database_url=UNREACHABLE_DB,
        redis_url=UNREACHABLE_REDIS,
        storage_backend="local",
        local_storage_root=tmp_path / "storage",
        health_timeout_seconds=2.0,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
