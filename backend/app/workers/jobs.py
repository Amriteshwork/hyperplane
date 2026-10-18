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
