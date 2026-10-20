"""Regression metrics (R², RMSE, MAE) and predicted-vs-actual / residual plot data."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.ml.evaluation.models import Evaluation, Metric, ScatterPlot

#: Charts get at most this many points (a fixed random subset), to keep payloads small.
MAX_POINTS = 1_000


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> Evaluation:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    shown = _subset(len(actual))
    return Evaluation(
        metrics=[
            Metric(
                id="r2",
                label="R²",
                value=float(r2_score(actual, predicted)),
                description="Share of the target's variance explained; 1 is perfect, 0 is "
                "no better than predicting the mean.",
                higher_is_better=True,
            ),
            Metric(
                id="rmse",
                label="RMSE",
                value=float(mean_squared_error(actual, predicted) ** 0.5),
                description="Typical error in the target's units; large misses count extra.",
                higher_is_better=False,
            ),
            Metric(
                id="mae",
                label="MAE",
                value=float(mean_absolute_error(actual, predicted)),
                description="Average absolute error in the target's units.",
                higher_is_better=False,
            ),
        ],
        plots=[
            ScatterPlot(
                id="predicted-vs-actual",
                title="Predicted vs actual",
                description="Each dot is a test row. Dots on the diagonal are perfect predictions.",
                x_label="Actual",
                y_label="Predicted",
                points=_points(actual[shown], predicted[shown]),
                reference="identity",
            ),
            ScatterPlot(
                id="residuals",
                title="Residuals",
                description="Error (actual − predicted) against the prediction. A shapeless "
                "band around zero is good; curves or funnels mean the model misses a pattern.",
                x_label="Predicted",
                y_label="Residual",
                points=_points(predicted[shown], residuals[shown]),
                reference="zero",
            ),
        ],
    )


def _subset(n: int) -> NDArray[np.intp]:
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, MAX_POINTS, replace=False))


def _points(x: NDArray[np.float64], y: NDArray[np.float64]) -> list[tuple[float, float]]:
    return [(_round(a), _round(b)) for a, b in zip(x, y, strict=True)]


def _round(value: float) -> float:
    return float(f"{value:.6g}")
