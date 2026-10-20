"""Regression metrics (R², RMSE, MAE) and predicted-vs-actual / residual plot data."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.ml.evaluation.models import Evaluation, Metric, Predictions, ScatterPlot
from app.ml.evaluation.numbers import points, subset


def evaluate_regression(predictions: Predictions) -> Evaluation:
    actual = np.asarray(predictions.y_true, dtype=float)
    predicted = np.asarray(predictions.y_pred, dtype=float)
    residuals = actual - predicted
    shown = subset(len(actual))
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
                points=points(actual[shown], predicted[shown]),
                reference="identity",
            ),
            ScatterPlot(
                id="residuals",
                title="Residuals",
                description="Error (actual − predicted) against the prediction. A shapeless "
                "band around zero is good; curves or funnels mean the model misses a pattern.",
                x_label="Predicted",
                y_label="Residual",
                points=points(predicted[shown], residuals[shown]),
                reference="zero",
            ),
        ],
    )
