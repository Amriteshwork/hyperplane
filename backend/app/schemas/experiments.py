"""Experiment/run API contract."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.domain.experiments import RunStatus, RunView
from app.domain.run_events import RunEventType
from app.ml.preprocessing import FeatureRole
from app.ml.profiling import SemanticType
from app.ml.training import TrainingReport
from app.ml.types import TaskType


class SplitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Fraction held out.")


class ExperimentIn(BaseModel):
    """Train one algorithm on a profiled dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID
    target: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    algorithm_id: str
    #: Hyperparameters, validated against the algorithm's `params_schema`; omitted = defaults.
    params: dict[str, JsonValue] = {}
    split: SplitIn = SplitIn()
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class ExperimentOut(BaseModel):
    id: UUID
    dataset_id: UUID
    target: str
    task_type: TaskType
    numeric_features: list[str]
    categorical_features: list[str]
    test_size: float
    seed: int


class RunOut(BaseModel):
    id: UUID
    status: RunStatus
    algorithm_id: str
    algorithm_version: str
    params: dict[str, JsonValue]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    experiment: ExperimentOut
    #: Metrics, plot data and coefficients once the run has succeeded.
    report: TrainingReport | None

    @classmethod
    def from_domain(cls, view: RunView) -> RunOut:
        run, experiment = view.run, view.experiment
        return cls(
            id=run.id,
            status=run.status,
            algorithm_id=run.algorithm_id,
            algorithm_version=run.algorithm_version,
            params=run.params,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error=run.error,
            experiment=ExperimentOut(
                id=experiment.id,
                dataset_id=experiment.dataset_id,
                target=experiment.target,
                task_type=experiment.task_type,
                numeric_features=list(experiment.preprocessing.numeric),
                categorical_features=list(experiment.preprocessing.categorical),
                test_size=experiment.test_size,
                seed=experiment.seed,
            ),
            report=run.report,
        )


class RunEventOut(BaseModel):
    """One SSE `data:` payload. `done`/`error` is always the last event of a stream."""

    type: RunEventType
    status: RunStatus | None = None
    step: int | None = None
    total: int | None = None
    message: str | None = None


class AlgorithmOptionOut(BaseModel):
    id: str
    name: str


class TargetOptionOut(BaseModel):
    column: str
    task_type: TaskType
    reason: str
    #: Algorithms that can be trained for this target now; empty if none support its task yet.
    algorithms: list[AlgorithmOptionOut]


class FeatureOptionOut(BaseModel):
    column: str
    semantic_type: SemanticType
    #: How the column enters the model; null if its type can't be a feature yet.
    role: FeatureRole | None


class TrainingOptionsOut(BaseModel):
    """What a dataset can be trained on: candidate targets (best first) and usable features."""

    targets: list[TargetOptionOut]
    features: list[FeatureOptionOut]
