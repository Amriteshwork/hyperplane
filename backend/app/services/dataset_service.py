"""Dataset use cases: upload, create from a sample, read metadata, profile and preview."""

import math
from dataclasses import dataclass, replace
from datetime import timedelta
from uuid import UUID, uuid4

import anyio.to_thread
import numpy as np
import pandas as pd
import structlog

from app.core.clock import Clock
from app.core.errors import (
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnprocessableError,
    UnsupportedMediaTypeError,
)
from app.db.repositories.base import DatasetRepository
from app.domain.datasets import (
    Cell,
    Dataset,
    DatasetSource,
    DatasetStatus,
    FileFormat,
    TablePreview,
)
from app.ingest.errors import IngestError, UnsupportedFormatError
from app.ingest.parquet import read_head, to_parquet_bytes
from app.ingest.readers import TableLimits, read_table
from app.ingest.sniff import sniff_format
from app.jobs.base import JobQueue
from app.jobs.names import PROFILE_DATASET
from app.ml.profiling import DatasetProfile
from app.samples.catalog import SampleDataset
from app.storage.base import StorageBackend

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UploadPolicy:
    max_bytes: int
    limits: TableLimits
    retention: timedelta


class DatasetService:
    def __init__(
        self,
        datasets: DatasetRepository,
        storage: StorageBackend,
        queue: JobQueue,
        clock: Clock,
        policy: UploadPolicy,
    ) -> None:
        self._datasets = datasets
        self._storage = storage
        self._queue = queue
        self._clock = clock
        self._policy = policy

    @property
    def max_upload_bytes(self) -> int:
        return self._policy.max_bytes

    async def upload(self, session_id: UUID, filename: str | None, data: bytes) -> Dataset:
        """Validate, store raw + Parquet, and queue profiling."""
        if len(data) > self._policy.max_bytes:
            raise PayloadTooLargeError(
                f"The file is larger than {self._policy.max_bytes // 2**20} MB."
            )
        name = (filename or "dataset").replace("\\", "/").rsplit("/", 1)[-1][:255] or "dataset"
        return await self._ingest(session_id, name, data, DatasetSource.UPLOAD, None)

    async def create_from_sample(self, session_id: UUID, sample: SampleDataset) -> Dataset:
        if sample.status != "available":
            raise ConflictError(f"Sample {sample.meta.id!r} has not been fetched on this server.")
        data = await anyio.to_thread.run_sync(sample.path.read_bytes)
        return await self._ingest(
            session_id, sample.meta.name, data, DatasetSource.SAMPLE, sample.meta.id
        )

    async def get(self, session_id: UUID, dataset_id: UUID) -> Dataset:
        """The dataset, if it belongs to this session and hasn't expired."""
        dataset = await self._datasets.get(dataset_id)
        if (
            dataset is None
            or dataset.session_id != session_id
            or dataset.expires_at <= self._clock.now()
        ):
            raise NotFoundError("Dataset not found.")
        return dataset

    async def get_profile(self, session_id: UUID, dataset_id: UUID) -> DatasetProfile:
        dataset = await self.get(session_id, dataset_id)
        if dataset.status is DatasetStatus.FAILED:
            raise ConflictError(f"Profiling failed: {dataset.error}")
        if dataset.profile is None:
            raise ConflictError("The dataset is still being profiled; try again shortly.")
        return dataset.profile

    async def preview(self, session_id: UUID, dataset_id: UUID, rows: int) -> TablePreview:
        dataset = await self.get(session_id, dataset_id)
        data = await anyio.to_thread.run_sync(self._storage.read, dataset.parquet_key)
        head = await anyio.to_thread.run_sync(read_head, data, rows)
        return TablePreview(
            columns=[str(c) for c in head.columns],
            rows=[[_cell(v) for v in row] for row in head.itertuples(index=False, name=None)],
            total_rows=dataset.n_rows,
        )

    async def _ingest(
        self,
        session_id: UUID,
        name: str,
        data: bytes,
        source: DatasetSource,
        sample_id: str | None,
    ) -> Dataset:
        fmt, df, parquet = await anyio.to_thread.run_sync(self._parse, data)
        dataset_id = uuid4()
        raw_key = f"datasets/{dataset_id}/raw.{fmt}"
        parquet_key = f"datasets/{dataset_id}/data.parquet"
        await anyio.to_thread.run_sync(self._storage.write, raw_key, data)
        await anyio.to_thread.run_sync(self._storage.write, parquet_key, parquet)

        now = self._clock.now()
        dataset = Dataset(
            id=dataset_id,
            session_id=session_id,
            name=name,
            source=source,
            sample_id=sample_id,
            format=fmt,
            size_bytes=len(data),
            raw_key=raw_key,
            parquet_key=parquet_key,
            n_rows=len(df),
            n_columns=df.shape[1],
            status=DatasetStatus.PROFILING,
            profile=None,
            error=None,
            created_at=now,
            expires_at=now + self._policy.retention,
        )
        await self._datasets.add(dataset)
        try:
            self._queue.enqueue(PROFILE_DATASET, dataset_id=str(dataset_id))
        except Exception as exc:
            log.error("profiling_enqueue_failed", dataset_id=str(dataset_id), error=repr(exc))
            failed = replace(
                dataset, status=DatasetStatus.FAILED, error="Could not queue profiling."
            )
            await self._datasets.update(failed)
            raise ServiceUnavailableError("The job queue is unavailable; try again.") from exc
        log.info("dataset_ingested", dataset_id=str(dataset_id), format=fmt, rows=len(df))
        return dataset

    def _parse(self, data: bytes) -> tuple[FileFormat, pd.DataFrame, bytes]:
        try:
            fmt = sniff_format(data)
            df = read_table(data, fmt, self._policy.limits)
        except UnsupportedFormatError as exc:
            raise UnsupportedMediaTypeError(str(exc)) from None
        except IngestError as exc:
            raise UnprocessableError(str(exc)) from None
        return fmt, df, to_parquet_bytes(df)


def _cell(value: object) -> Cell:
    """JSON-safe preview cell."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return str(value.isoformat())
    if isinstance(value, str):
        return value
    return None if bool(pd.isna(value)) else str(value)
