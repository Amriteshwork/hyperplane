"""SQL repositories and profiling against a real, migrated Postgres.

Opt-in: HYPERPLANE_TEST_POSTGRES=1 with DATABASE_URL pointing at a database where
`alembic upgrade head` has run (CI does this; locally `make up`).
"""

import os
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models import SessionRow
from app.db.repositories.sql_datasets import SqlDatasetRepository
from app.db.repositories.sql_sessions import SqlSessionRepository
from app.domain.datasets import Dataset, DatasetSource, DatasetStatus, FileFormat
from app.domain.sessions import Session
from app.ingest.parquet import to_parquet_bytes
from app.services.profiling_service import ProfilingService
from tests.fakes.storage import InMemoryStorage

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.anyio,
    pytest.mark.skipif(
        os.environ.get("HYPERPLANE_TEST_POSTGRES") != "1", reason="HYPERPLANE_TEST_POSTGRES!=1"
    ),
]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(Settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    session_id = uuid4()
    async with factory() as db:
        db.info["session_id"] = session_id
        yield db
        # Deleting the session cascades to its datasets.
        await db.execute(delete(SessionRow).where(SessionRow.id == session_id))
        await db.commit()
    await engine.dispose()


def _dataset(session_id: object) -> Dataset:
    dataset_id = uuid4()
    return Dataset(
        id=dataset_id,
        session_id=session_id,  # type: ignore[arg-type]
        name="d.csv",
        source=DatasetSource.UPLOAD,
        sample_id=None,
        format=FileFormat.CSV,
        size_bytes=10,
        raw_key=f"datasets/{dataset_id}/raw.csv",
        parquet_key=f"datasets/{dataset_id}/data.parquet",
        n_rows=3,
        n_columns=2,
        status=DatasetStatus.PROFILING,
        profile=None,
        error=None,
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
    )


async def test_sessions_round_trip(db: AsyncSession) -> None:
    repo = SqlSessionRepository(db)
    session = Session(id=db.info["session_id"], created_at=NOW, last_seen=NOW)
    await repo.add(session)
    assert await repo.get(session.id) == session

    later = NOW + timedelta(hours=1)
    await repo.touch(session.id, later)
    stored = await repo.get(session.id)
    assert stored is not None and stored.last_seen == later
    assert await repo.get(uuid4()) is None


async def test_datasets_round_trip_and_profiling(db: AsyncSession) -> None:
    await SqlSessionRepository(db).add(
        Session(id=db.info["session_id"], created_at=NOW, last_seen=NOW)
    )
    repo = SqlDatasetRepository(db)
    dataset = _dataset(db.info["session_id"])
    await repo.add(dataset)
    assert await repo.get(dataset.id) == dataset

    storage = InMemoryStorage()
    storage.write(
        dataset.parquet_key,
        to_parquet_bytes(pd.DataFrame({"x": [1.0, 2.0, 3.5], "y": ["a", "b", "a"]})),
    )
    await ProfilingService(repo, storage).run(dataset.id)

    profiled = await repo.get(dataset.id)
    assert profiled is not None and profiled.status is DatasetStatus.READY
    assert profiled.profile is not None and [c.name for c in profiled.profile.columns] == ["x", "y"]

    with pytest.raises(LookupError):
        await repo.update(replace(dataset, id=uuid4()))
