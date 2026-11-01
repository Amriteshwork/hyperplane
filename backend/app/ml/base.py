"""Algorithm Pack contract (ARCHITECTURE §4, §12): shared types + capability protocols.

Required capabilities make up `AlgorithmPack`. Optional ones (`Reports*`) are checked with
`isinstance` and declared in `AlgorithmSpec.supports`; packs never stub them out.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from app.ml.profiling.profile import DatasetProfile
from app.ml.types import TaskType

__all__ = ["TaskType"]  # re-exported: packs import it from app.ml.base


class Capability(StrEnum):
    PREDICT_PROBA = "predict_proba"
    FEATURE_IMPORTANCE = "feature_importance"
    COEFFICIENTS = "coefficients"
    CLASS_COEFFICIENTS = "class_coefficients"
    TRAINING_CURVE = "training_curve"
    TUNING_CURVE = "tuning_curve"
    NETWORK = "network"
    NEIGHBOURS = "neighbours"
    CLASS_GAUSSIANS = "class_gaussians"
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
    #: The estimator rejects sparse matrices, so preprocessing must always return a dense one.
    dense_input: bool = False
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


@dataclass(frozen=True, slots=True)
class TrainingCurve:
    """What an iterative model recorded once per epoch, in order."""

    loss: Sequence[float]
    #: The held-out score per epoch, when the model stopped early on one.
    validation: Sequence[float] | None = None
    #: What `validation` measures, e.g. "Validation accuracy".
    validation_label: str = "Validation score"


@dataclass(frozen=True, slots=True)
class TuningCurve:
    """A held-out score for several values of one hyperparameter, in increasing order."""

    #: Its name in the params model, so the Lab can fill in the best value, e.g. "n_neighbors".
    name: str
    #: The hyperparameter on the x axis, e.g. "Neighbours (k)".
    parameter: str
    values: Sequence[float]
    scores: Sequence[float]
    #: What `scores` measures, e.g. "Cross-validated accuracy".
    score_label: str
    #: The value this run used, when it is among `values`.
    chosen: float | None = None
    #: Values spanning several orders of magnitude read better on a log axis.
    log_scale: bool = False


@dataclass(frozen=True, slots=True)
class Network:
    """A fitted feed-forward network: its units, layer by layer, and the weights between them."""

    input_names: Sequence[str]
    #: `weights[i]` has a row per unit of layer i and a column per unit of layer i + 1.
    weights: Sequence[NDArray[Any]]
    #: The hidden layers' activation, e.g. "relu".
    activation: str
    output_activation: str
    output_names: Sequence[str]


class Neighbour(BaseModel):
    """A stored training row that had a say in one prediction."""

    model_config = ConfigDict(frozen=True)

    distance: float
    #: This row's share of the say: 1/k each, unless closer rows count more.
    share: float
    target: str
    #: Its values in `NeighbourReport.columns`, as the learner's data shows them.
    values: list[str]


class NeighbourExample(BaseModel):
    """One test row and the training rows that decided its prediction."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["miss", "hit"]
    predicted: str
    actual: str
    values: list[str]
    neighbours: list[Neighbour]


class NeighbourReport(BaseModel):
    """What a stored-rows model did: some predictions traced to rows, and how near "near" is."""

    model_config = ConfigDict(frozen=True)

    k: int
    stored_rows: int
    #: The columns where neighbours differ most from their query (at most a handful).
    columns: list[str]
    total_columns: int
    examples: list[NeighbourExample]
    #: On a sample of test rows: the mean distance to the k-th neighbour, and to any stored row.
    nearest_distance: float
    typical_distance: float


class FeatureGaussians(BaseModel):
    """One feature's bell curve in each class, in the data's own units."""

    model_config = ConfigDict(frozen=True)

    feature: str
    #: One per class, in the order of `GaussianReport.classes`.
    means: list[float]
    stds: list[float]
    #: Variance of the class means over the average within-class variance.
    separation: float


class Evidence(BaseModel):
    """What one feature adds to the log score of the predicted class and of its rival."""

    model_config = ConfigDict(frozen=True)

    feature: str
    #: The value the model saw, after imputation: missing values show what they were filled with.
    value: str
    predicted: float
    rival: float


class GaussianExample(BaseModel):
    """One test row's prediction, added up from the prior and each feature's log likelihood."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["miss", "hit"]
    predicted: str
    actual: str
    #: The class being compared: the actual one for a miss, otherwise the runner-up.
    rival: str
    #: The model's probability for `predicted`.
    confidence: float
    prior: Evidence
    #: The features that separate the two classes most for this row, largest gap first.
    evidence: list[Evidence]
    #: The sum over every other feature, so the column totals are the real log scores.
    others: Evidence | None


class CorrelatedPair(BaseModel):
    """The numeric columns most correlated within a class, which the model treats as independent."""

    model_config = ConfigDict(frozen=True)

    a: str
    b: str
    correlation: float


class GaussianReport(BaseModel):
    """What a per-class Gaussian model learned, how it decides, and how far to trust its odds."""

    model_config = ConfigDict(frozen=True)

    classes: list[str]
    priors: list[float]
    #: The features that separate the classes most; at most a handful.
    features: list[FeatureGaussians]
    total_features: int
    examples: list[GaussianExample]
    mean_confidence: float
    accuracy: float
    correlated: CorrelatedPair | None


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
class ReportsTrainingCurve(Protocol):
    """Models trained epoch by epoch that keep their loss per epoch (e.g. `loss_curve_`).

    `None` when this run's settings recorded no curve (e.g. a full-batch solver).
    """

    def training_curve(self, fitted: Estimator) -> TrainingCurve | None: ...


@runtime_checkable
class ReportsTuningCurve(Protocol):
    """Models whose key hyperparameter can be rescored cheaply on the training rows.

    Takes the whole pipeline, so each fold refits the preprocessing. `None` when this dataset
    would make the view too slow.
    """

    def tuning_curve(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: ArrayLike
    ) -> TuningCurve | None: ...


@runtime_checkable
class ReportsNeighbours(Protocol):
    """Models that answer from stored rows, which can show the rows behind a prediction."""

    def neighbours(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> NeighbourReport: ...


@runtime_checkable
class ReportsClassGaussians(Protocol):
    """Models that fit a bell curve per class and feature, which can show them and add them up."""

    def class_gaussians(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> GaussianReport: ...


@runtime_checkable
class ReportsNetwork(Protocol):
    """Neural networks, whose layers and weights the Lab draws."""

    def network(self, fitted: Estimator, feature_names: list[str]) -> Network: ...


@runtime_checkable
class ReportsIterations(Protocol):
    def iteration_callback(self, publish: ProgressFn) -> IterationCallback: ...
