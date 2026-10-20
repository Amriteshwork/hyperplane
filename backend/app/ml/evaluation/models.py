"""What evaluation reports: metrics a learner can read and plot data for the charts (JSON-safe)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


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


class Evaluation(_Frozen):
    metrics: list[Metric]
    plots: list[ScatterPlot]
