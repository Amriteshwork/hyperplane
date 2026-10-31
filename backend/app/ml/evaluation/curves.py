"""Plots of what a model recorded while training (a loss per epoch) or tuning (a score per k)."""

from collections.abc import Sequence

import numpy as np

from app.ml.base import TrainingCurve, TuningCurve
from app.ml.evaluation.models import CurveMarker, CurvePlot, CurveSeries, TuningSuggestion
from app.ml.evaluation.numbers import evenly, points, round6


def training_curve_plots(curve: TrainingCurve) -> list[CurvePlot]:
    marker = None
    if curve.validation is not None:
        # scikit-learn keeps the weights of the first epoch with the best validation score.
        best = int(np.argmax(curve.validation)) + 1
        marker = CurveMarker(x=best, label=f"Weights kept (epoch {best})")
    plots = [
        _per_epoch(
            "loss-curve",
            "Training loss per epoch",
            "The average loss on the training rows after each pass over them. It should fall "
            "quickly, then flatten; training stops once it stops improving.",
            "Training loss",
            curve.loss,
            marker,
        )
    ]
    if curve.validation is not None:
        plots.append(
            _per_epoch(
                "validation-curve",
                f"{curve.validation_label} per epoch",
                "The score on rows held out of training (early stopping). Training stopped "
                "when it stopped improving, and kept the weights from its best epoch.",
                curve.validation_label,
                curve.validation,
                marker,
            )
        )
    return plots


def tuning_curve_plot(curve: TuningCurve) -> CurvePlot:
    x, y = np.asarray(curve.values, dtype=float), np.asarray(curve.scores, dtype=float)
    return CurvePlot(
        id="tuning-curve",
        title=f"{curve.score_label} by {curve.parameter}",
        description=(
            "Each point refits on parts of the training rows and scores on the part left out, "
            "so you can compare settings without touching the test set. Pick the best-scoring "
            "region, then rerun with that value."
        ),
        x_label=curve.parameter,
        y_label=curve.score_label,
        series=[CurveSeries(name=curve.score_label, points=points(x, y))],
        axes="data",
        marker=None
        if curve.chosen is None
        else CurveMarker(x=curve.chosen, label=f"This run ({curve.chosen:g})"),
    )


def tuning_suggestion(curve: TuningCurve) -> TuningSuggestion:
    best = int(np.argmax(curve.scores))  # the first of equal scores
    return TuningSuggestion(
        name=curve.name,
        label=curve.parameter,
        best=curve.values[best],
        score=round6(curve.scores[best]),
        score_label=curve.score_label,
        chosen=curve.chosen,
    )


def _per_epoch(
    plot_id: str,
    title: str,
    description: str,
    label: str,
    values: Sequence[float],
    marker: CurveMarker | None,
) -> CurvePlot:
    y = np.asarray(values, dtype=float)
    keep = evenly(len(y))
    return CurvePlot(
        id=plot_id,
        title=title,
        description=description,
        x_label="Epoch",
        y_label=label,
        series=[CurveSeries(name=label, points=points((keep + 1).astype(float), y[keep]))],
        axes="data",
        marker=marker,
    )
