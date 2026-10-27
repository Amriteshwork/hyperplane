"""Metrics and plot data per task type (EVALUATORS[task_type])."""

from collections.abc import Callable

from app.ml.evaluation.classification import evaluate_classification
from app.ml.evaluation.curves import training_curve_plots
from app.ml.evaluation.models import (
    CurveMarker,
    CurvePlot,
    CurveSeries,
    Evaluation,
    HeatmapPlot,
    Metric,
    NetworkLayer,
    NetworkSummary,
    Plot,
    Predictions,
    ScatterPlot,
)
from app.ml.evaluation.network import summarise_network
from app.ml.evaluation.regression import evaluate_regression
from app.ml.types import TaskType

type Evaluator = Callable[[Predictions], Evaluation]

#: Task types that can be trained are exactly the ones with an evaluator.
EVALUATORS: dict[TaskType, Evaluator] = {
    TaskType.REGRESSION: evaluate_regression,
    TaskType.BINARY: evaluate_classification,
    TaskType.MULTICLASS: evaluate_classification,
}

__all__ = [
    "EVALUATORS",
    "CurveMarker",
    "CurvePlot",
    "CurveSeries",
    "Evaluation",
    "Evaluator",
    "HeatmapPlot",
    "Metric",
    "NetworkLayer",
    "NetworkSummary",
    "Plot",
    "Predictions",
    "ScatterPlot",
    "summarise_network",
    "training_curve_plots",
]
