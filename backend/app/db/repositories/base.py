"""Repository protocols services depend on. `Sql*` implementations live alongside."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.datasets import Dataset
from app.domain.sessions import Session


class SessionRepository(Protocol):
    async def get(self, session_id: UUID) -> Session | None: ...

    async def add(self, session: Session) -> None: ...

    async def touch(self, session_id: UUID, when: datetime) -> None:
        """Update `last_seen`."""
        ...


class DatasetRepository(Protocol):
    async def add(self, dataset: Dataset) -> None: ...

    async def get(self, dataset_id: UUID) -> Dataset | None: ...

    async def update(self, dataset: Dataset) -> None:
        """Persist every field of an existing dataset."""
        ...


class RunRepository(Protocol):
    """Persistence for experiments and their runs. Methods arrive with the training pipeline."""
