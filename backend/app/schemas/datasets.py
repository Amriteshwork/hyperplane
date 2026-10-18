"""Dataset API contract."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.domain.datasets import Dataset, DatasetSource, DatasetStatus, FileFormat, TablePreview


class DatasetOut(BaseModel):
    id: UUID
    name: str
    source: DatasetSource
    sample_id: str | None
    format: FileFormat
    size_bytes: int
    n_rows: int
    n_columns: int
    status: DatasetStatus
    error: str | None
    created_at: datetime
    expires_at: datetime

    @classmethod
    def from_domain(cls, dataset: Dataset) -> DatasetOut:
        return cls(
            id=dataset.id,
            name=dataset.name,
            source=dataset.source,
            sample_id=dataset.sample_id,
            format=dataset.format,
            size_bytes=dataset.size_bytes,
            n_rows=dataset.n_rows,
            n_columns=dataset.n_columns,
            status=dataset.status,
            error=dataset.error,
            created_at=dataset.created_at,
            expires_at=dataset.expires_at,
        )


class DatasetPreviewOut(BaseModel):
    columns: list[str]
    rows: list[list[str | int | float | bool | None]]
    total_rows: int

    @classmethod
    def from_domain(cls, preview: TablePreview) -> DatasetPreviewOut:
        return cls(columns=preview.columns, rows=preview.rows, total_rows=preview.total_rows)
