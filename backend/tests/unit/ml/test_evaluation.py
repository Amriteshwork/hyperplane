import numpy as np
import pytest

from app.ml.evaluation import EVALUATORS
from app.ml.evaluation.regression import MAX_POINTS, evaluate_regression
from app.ml.types import TaskType


def test_regression_metrics() -> None:
    evaluation = evaluate_regression([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    metrics = {m.id: m.value for m in evaluation.metrics}

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(1 - 4 / 5)


def test_regression_plots_pair_points_and_draw_guides() -> None:
    evaluation = evaluate_regression([1.0, 2.0], [1.5, 2.0])
    plots = {p.id: p for p in evaluation.plots}

    assert plots["predicted-vs-actual"].points == [(1.0, 1.5), (2.0, 2.0)]
    assert plots["predicted-vs-actual"].reference == "identity"
    assert plots["residuals"].points == [(1.5, -0.5), (2.0, 0.0)]
    assert plots["residuals"].reference == "zero"


def test_plots_are_subsampled_deterministically() -> None:
    y = np.arange(5_000, dtype=float)
    first = evaluate_regression(y, y + 1).plots[0].points
    assert len(first) == MAX_POINTS
    assert first == evaluate_regression(y, y + 1).plots[0].points


def test_regression_is_registered() -> None:
    assert EVALUATORS[TaskType.REGRESSION] is evaluate_regression
