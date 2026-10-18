"""Dict-backed repositories with the same behaviour as the SQL ones."""

from dataclasses import replace
from datetime import datetime
from uuid import UUID

from app.domain.datasets import Dataset
from app.domain.sessions import Session


class InMemorySessionRepository:
    def __init__(self) -> None:
        self.rows: dict[UUID, Session] = {}

    async def get(self, session_id: UUID) -> Session | None:
        return self.rows.get(session_id)

    async def add(self, session: Session) -> None:
        self.rows[session.id] = session

    async def touch(self, session_id: UUID, when: datetime) -> None:
        if session_id in self.rows:
            self.rows[session_id] = replace(self.rows[session_id], last_seen=when)


class InMemoryDatasetRepository:
    def __init__(self) -> None:
        self.rows: dict[UUID, Dataset] = {}

    async def add(self, dataset: Dataset) -> None:
        self.rows[dataset.id] = dataset

    async def get(self, dataset_id: UUID) -> Dataset | None:
        return self.rows.get(dataset_id)

    async def update(self, dataset: Dataset) -> None:
        if dataset.id not in self.rows:
            raise LookupError(f"dataset {dataset.id} does not exist")
        self.rows[dataset.id] = dataset
