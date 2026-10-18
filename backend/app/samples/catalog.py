"""Read-only index of the sample datasets in `DATASETS_DIR`."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.ml.types import TaskType

DATA_FILE = "data.csv"
META_FILE = "meta.yaml"


class SampleMeta(BaseModel):
    """Schema of `meta.yaml` (see datasets/README.md)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    name: str
    description: str
    task: TaskType
    target: str
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    vendored: bool
    licence: str
    source: str
    url: str | None
    citation: str | None


class SampleDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    meta: SampleMeta
    path: Path
    status: Literal["available", "not-fetched"]


class UnknownSampleError(KeyError):
    """No sample with this id."""


class SampleCatalog:
    def __init__(self, samples: Sequence[SampleDataset]) -> None:
        self._samples = {s.meta.id: s for s in samples}

    @classmethod
    def load(cls, root: Path) -> SampleCatalog:
        """Index every `<root>/<id>/meta.yaml`; a missing root gives an empty catalog."""
        samples = []
        for meta_path in sorted(root.glob(f"*/{META_FILE}")) if root.is_dir() else []:
            meta = SampleMeta.model_validate(yaml.safe_load(meta_path.read_text("utf-8")))
            if meta.id != meta_path.parent.name:
                raise ValueError(f"{meta_path}: id {meta.id!r} must match its folder name")
            data_path = meta_path.parent / DATA_FILE
            status: Literal["available", "not-fetched"] = (
                "available" if data_path.is_file() else "not-fetched"
            )
            samples.append(SampleDataset(meta=meta, path=data_path, status=status))
        return cls(samples)

    def list(self) -> Sequence[SampleDataset]:
        return list(self._samples.values())

    def get(self, sample_id: str) -> SampleDataset:
        try:
            return self._samples[sample_id]
        except KeyError:
            raise UnknownSampleError(sample_id) from None
