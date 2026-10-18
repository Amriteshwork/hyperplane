"""Datasets: an uploaded or sample table, its stored files and its profile."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.ml.profiling import DatasetProfile


class DatasetSource(StrEnum):
    UPLOAD = "upload"
    SAMPLE = "sample"


class DatasetStatus(StrEnum):
    PROFILING = "profiling"
    READY = "ready"
    FAILED = "failed"


class FileFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    XLSX = "xlsx"


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    id: UUID
    session_id: UUID
    name: str
    source: DatasetSource
    sample_id: str | None
    format: FileFormat
    size_bytes: int
    raw_key: str
    parquet_key: str
    n_rows: int
    n_columns: int
    status: DatasetStatus
    profile: DatasetProfile | None
    error: str | None
    created_at: datetime
    expires_at: datetime


type Cell = str | int | float | bool | None


@dataclass(frozen=True, slots=True, kw_only=True)
class TablePreview:
    columns: list[str]
    rows: list[list[Cell]]
    total_rows: int
