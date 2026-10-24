"""Metrics and plot data per task type (EVALUATORS[task_type])."""

from collections.abc import Callable

from app.ml.evaluation.classification import evaluate_classification
from app.ml.evaluation.curves import training_curve_plots
from app.ml.evaluation.models import (
    CurvePlot,
    CurveSeries,
    Evaluation,
    HeatmapPlot,
    Metric,
    Plot,
    Predictions,
    ScatterPlot,
)
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
    "CurvePlot",
    "CurveSeries",
    "Evaluation",
    "Evaluator",
    "HeatmapPlot",
    "Metric",
    "Plot",
    "Predictions",
    "ScatterPlot",
    "training_curve_plots",
]
