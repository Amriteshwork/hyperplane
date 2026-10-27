"""What evaluation reports: metrics a learner can read and plot data for the charts (JSON-safe)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class Predictions:
    """What an evaluator scores: the test targets and the model's outputs for them."""

    y_true: ArrayLike
    y_pred: ArrayLike
    #: Class probabilities, one column per entry of `classes`; classifiers that have them only.
    proba: NDArray[Any] | None = None
    #: The model's classes in its own (sorted) order; empty for regression.
    classes: Sequence[Any] = ()


class Metric(_Frozen):
    id: str
    label: str
    value: float
    description: str
    higher_is_better: bool


class ScatterPlot(_Frozen):
    kind: Literal["scatter"] = "scatter"
    id: str
    title: str
    description: str
    x_label: str
    y_label: str
    points: list[tuple[float, float]]
    #: A guide line to draw: `identity` is y = x (perfect predictions), `zero` is y = 0.
    reference: Literal["identity", "zero"] | None = None


class HeatmapPlot(_Frozen):
    """A grid of numbers, e.g. a confusion matrix."""

    kind: Literal["heatmap"] = "heatmap"
    id: str
    title: str
    description: str
    x_label: str
    y_label: str
    x_categories: list[str]
    y_categories: list[str]
    #: `values[i][j]` is the cell in row `y_categories[i]`, column `x_categories[j]`.
    values: list[list[float]]


class CurveSeries(_Frozen):
    name: str
    points: list[tuple[float, float]]


class CurveMarker(_Frozen):
    """A labelled vertical guide, e.g. the epoch whose weights early stopping kept."""

    x: float
    label: str


class CurvePlot(_Frozen):
    """One or more lines on shared axes, e.g. ROC curves or a loss per epoch."""

    kind: Literal["curve"] = "curve"
    id: str
    title: str
    description: str
    x_label: str
    y_label: str
    series: list[CurveSeries]
    #: `diagonal` is y = x from (0, 0) to (1, 1): a random classifier's ROC curve.
    reference: Literal["diagonal"] | None = None
    #: `unit`: both axes run 0–1 (rates, e.g. ROC); `data`: axes fit the points (e.g. a loss).
    axes: Literal["unit", "data"] = "unit"
    marker: CurveMarker | None = None


Plot = Annotated[ScatterPlot | HeatmapPlot | CurvePlot, Field(discriminator="kind")]


class Evaluation(_Frozen):
    metrics: list[Metric]
    plots: list[Plot]


class NetworkLayer(_Frozen):
    kind: Literal["input", "hidden", "output"]
    units: int
    #: None for the inputs, which pass their values on unchanged.
    activation: str | None
    #: Labels of the units drawn (at most `MAX_SHOWN`): feature or class names, else "neuron k".
    shown: list[str]


class NetworkSummary(_Frozen):
    """A fitted network to draw: every layer's size, and the weights between the units shown."""

    layers: list[NetworkLayer]
    #: Weights and biases, all layers.
    parameters: int
    #: `weights[i][a][b]` joins shown unit a of layer i to shown unit b of layer i + 1.
    weights: list[list[list[float]]]
