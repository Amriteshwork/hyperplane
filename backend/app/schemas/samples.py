"""Sample dataset API contract."""

from typing import Literal

from pydantic import BaseModel

from app.ml.types import TaskType
from app.samples.catalog import SampleDataset


class SampleOut(BaseModel):
    id: str
    name: str
    description: str
    task: TaskType
    target: str
    rows: int
    columns: int
    licence: str
    source: str
    url: str | None
    citation: str | None
    status: Literal["available", "not-fetched"]

    @classmethod
    def from_domain(cls, sample: SampleDataset) -> SampleOut:
        meta = sample.meta
        return cls(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            task=meta.task,
            target=meta.target,
            rows=meta.rows,
            columns=meta.columns,
            licence=meta.licence,
            source=meta.source,
            url=meta.url,
            citation=meta.citation,
            status=sample.status,
        )
