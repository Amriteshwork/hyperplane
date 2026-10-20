"""RQ job functions, enqueued by the dotted paths in `app.jobs.names`. Thin glue only."""

import asyncio
from uuid import UUID

import structlog

from app.workers.bootstrap import container


def profile_dataset(dataset_id: str) -> None:
    """Profile a stored dataset and save the result on its row."""
    with structlog.contextvars.bound_contextvars(job="profile_dataset", dataset_id=dataset_id):
        asyncio.run(_profile_dataset(UUID(dataset_id)))


async def _profile_dataset(dataset_id: UUID) -> None:
    async with container().profiling_service() as service:
        await service.run(dataset_id)


def train_run(run_id: str) -> None:
    """Train a queued run, store its model and report, and publish progress."""
    with structlog.contextvars.bound_contextvars(job="train_run", run_id=run_id):
        asyncio.run(_train_run(UUID(run_id)))


async def _train_run(run_id: UUID) -> None:
    async with container().training_service() as service:
        await service.run(run_id)
