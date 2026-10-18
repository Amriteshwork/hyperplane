from dataclasses import asdict
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DatasetRow
from app.domain.datasets import Dataset, DatasetSource, DatasetStatus, FileFormat
from app.ml.profiling import DatasetProfile


class SqlDatasetRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def add(self, dataset: Dataset) -> None:
        self._db.add(DatasetRow(**_to_columns(dataset)))
        await self._db.commit()

    async def get(self, dataset_id: UUID) -> Dataset | None:
        row = await self._db.get(DatasetRow, dataset_id, populate_existing=True)
        return _to_domain(row) if row is not None else None

    async def update(self, dataset: Dataset) -> None:
        row = await self._db.get(DatasetRow, dataset.id)
        if row is None:
            raise LookupError(f"dataset {dataset.id} does not exist")
        for column, value in _to_columns(dataset).items():
            setattr(row, column, value)
        await self._db.commit()


def _to_columns(dataset: Dataset) -> dict[str, Any]:
    columns = asdict(dataset)
    columns["profile"] = dataset.profile.model_dump(mode="json") if dataset.profile else None
    return columns


def _to_domain(row: DatasetRow) -> Dataset:
    return Dataset(
        id=row.id,
        session_id=row.session_id,
        name=row.name,
        source=DatasetSource(row.source),
        sample_id=row.sample_id,
        format=FileFormat(row.format),
        size_bytes=row.size_bytes,
        raw_key=row.raw_key,
        parquet_key=row.parquet_key,
        n_rows=row.n_rows,
        n_columns=row.n_columns,
        status=DatasetStatus(row.status),
        profile=DatasetProfile.model_validate(row.profile) if row.profile else None,
        error=row.error,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )
