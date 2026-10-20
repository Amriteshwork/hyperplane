"""Experiments (a training setup on one dataset) and runs (one training of one algorithm)."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import JsonValue

from app.ml.preprocessing import PreprocessingConfig
from app.ml.training import TrainingConfig, TrainingReport
from app.ml.types import TaskType


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED}


@dataclass(frozen=True, slots=True, kw_only=True)
class Experiment:
    id: UUID
    dataset_id: UUID
    target: str
    task_type: TaskType
    preprocessing: PreprocessingConfig
    test_size: float
    seed: int
    created_at: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    id: UUID
    experiment_id: UUID
    algorithm_id: str
    algorithm_version: str
    #: Validated hyperparameters with defaults filled in.
    params: dict[str, JsonValue]
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report: TrainingReport | None = None
    artifact_key: str | None = None
    error: str | None = None
    env: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RunView:
    """A run with the experiment it belongs to."""

    run: Run
    experiment: Experiment

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            target=self.experiment.target,
            task_type=self.experiment.task_type,
            preprocessing=self.experiment.preprocessing,
            test_size=self.experiment.test_size,
            seed=self.experiment.seed,
            params=self.run.params,
        )
