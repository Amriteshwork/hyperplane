"""Profiles a stored dataset. Runs in the worker (`app.workers.jobs.profile_dataset`)."""

from dataclasses import replace
from uuid import UUID

import structlog

from app.db.repositories.base import DatasetRepository
from app.domain.datasets import DatasetStatus
from app.ingest.parquet import read_parquet
from app.ml.profiling import profile_dataframe
from app.storage.base import BlobReader

log = structlog.get_logger(__name__)


class ProfilingService:
    def __init__(self, datasets: DatasetRepository, storage: BlobReader) -> None:
        self._datasets = datasets
        self._storage = storage

    async def run(self, dataset_id: UUID) -> None:
        """Profile the dataset and store the result; on failure mark it failed and re-raise."""
        dataset = await self._datasets.get(dataset_id)
        if dataset is None:
            log.warning("profiling_skipped_missing_dataset", dataset_id=str(dataset_id))
            return
        try:
            df = read_parquet(self._storage.read(dataset.parquet_key))
            profile = profile_dataframe(df)
        except Exception as exc:
            await self._datasets.update(
                replace(
                    dataset,
                    status=DatasetStatus.FAILED,
                    error=f"Profiling failed ({type(exc).__name__}).",
                )
            )
            raise
        await self._datasets.update(replace(dataset, status=DatasetStatus.READY, profile=profile))
        log.info("dataset_profiled", dataset_id=str(dataset_id), columns=profile.n_columns)
