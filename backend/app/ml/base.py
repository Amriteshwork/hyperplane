"""Algorithm Pack contract (ARCHITECTURE §4, §12): shared types + capability protocols.

Required capabilities make up `AlgorithmPack`. Optional ones (`Reports*`) are checked with
`isinstance` and declared in `AlgorithmSpec.supports`; packs never stub them out.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from app.ml.profiling.profile import DatasetProfile
from app.ml.types import TaskType

__all__ = ["TaskType"]  # re-exported: packs import it from app.ml.base


class Capability(StrEnum):
    PREDICT_PROBA = "predict_proba"
    FEATURE_IMPORTANCE = "feature_importance"
    COEFFICIENTS = "coefficients"
    CLASS_COEFFICIENTS = "class_coefficients"
    SPARSE = "sparse"


class AlgorithmSpec(BaseModel):
    """Pack metadata (`spec.py`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    family: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    task_types: frozenset[TaskType] = Field(min_length=1)
    difficulty: int = Field(ge=1, le=5)
    concepts: tuple[str, ...] = ()
    supports: frozenset[Capability] = frozenset()
    max_rows_default: int = Field(gt=0)
    content_path: str


class Suitability(BaseModel):
    """How well a pack fits a dataset, with reasons a learner can read."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()


class Annotation(BaseModel):
    """One `annotations.yaml` entry, keyed by its `# @a:<anchor>`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    short: str
    deep: str
    concepts: tuple[str, ...] = ()
    span: Literal["line", "block"] = "line"


@dataclass(frozen=True, slots=True)
class ExplainedTemplate:
    """A code template (with anchors) and its explanations."""

    source: str
    annotations: Mapping[str, Annotation] = field(default_factory=dict)
    filename: str = "train.py"
    #: Human label, e.g. "Training script" or "Under the hood".
    title: str = "Training script"
    #: Jinja (`.j2`) sources are rendered with a run's config; plain sources are shown as-is.
    templated: bool = False


class Progress(BaseModel):
    step: int = Field(ge=0)
    total: int | None = None
    message: str | None = None


type ProgressFn = Callable[[Progress], None]
type IterationCallback = Callable[[int], None]
type CoefTable = Mapping[str, float]
#: One `CoefTable` per class, keyed by the class label as text.
type ClassCoefTable = Mapping[str, CoefTable]
type ImportanceTable = Mapping[str, float]


class Estimator(Protocol):
    """The subset of the scikit-learn estimator API the platform relies on."""

    def fit(self, X: ArrayLike, y: ArrayLike, /) -> object: ...

    def predict(self, X: ArrayLike, /) -> NDArray[Any]: ...


# ── required capabilities ────────────────────────────────────────────────────


@runtime_checkable
class Describable(Protocol):
    spec: AlgorithmSpec


@runtime_checkable
class Parameterised(Protocol):
    params_model: type[BaseModel]


@runtime_checkable
class Buildable(Protocol):
    def build_estimator(self, params: BaseModel) -> Estimator: ...


@runtime_checkable
class Explainable(Protocol):
    def templates(self) -> Mapping[str, ExplainedTemplate]: ...


@runtime_checkable
class Recommendable(Protocol):
    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability: ...


@runtime_checkable
class AlgorithmPack(Describable, Parameterised, Buildable, Explainable, Recommendable, Protocol):
    """Everything a pack must provide."""


# ── optional capabilities ────────────────────────────────────────────────────


@runtime_checkable
class ReportsCoefficients(Protocol):
    def coefficients(self, fitted: Estimator, feature_names: list[str]) -> CoefTable: ...


@runtime_checkable
class ReportsClassCoefficients(Protocol):
    """Classifiers with a weight per (class, feature), e.g. one row of `coef_` per class."""

    def class_coefficients(self, fitted: Estimator, feature_names: list[str]) -> ClassCoefTable: ...


@runtime_checkable
class ReportsFeatureImportance(Protocol):
    def feature_importance(
        self, fitted: Estimator, feature_names: list[str]
    ) -> ImportanceTable: ...


@runtime_checkable
class ReportsIterations(Protocol):
    def iteration_callback(self, publish: ProgressFn) -> IterationCallback: ...
