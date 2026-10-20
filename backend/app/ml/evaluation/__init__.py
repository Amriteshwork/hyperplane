"""Metrics and plot data per task type (EVALUATORS[task_type])."""

from collections.abc import Callable

from numpy.typing import ArrayLike

from app.ml.evaluation.models import Evaluation, Metric, ScatterPlot
from app.ml.evaluation.regression import evaluate_regression
from app.ml.types import TaskType

type Evaluator = Callable[[ArrayLike, ArrayLike], Evaluation]

#: Task types that can be trained are exactly the ones with an evaluator.
EVALUATORS: dict[TaskType, Evaluator] = {
    TaskType.REGRESSION: evaluate_regression,
}

__all__ = ["EVALUATORS", "Evaluation", "Evaluator", "Metric", "ScatterPlot"]
