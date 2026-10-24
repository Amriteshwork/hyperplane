"""Plots of what an iterative model recorded while training (its loss per epoch)."""

from collections.abc import Sequence

import numpy as np

from app.ml.base import TrainingCurve
from app.ml.evaluation.models import CurvePlot, CurveSeries
from app.ml.evaluation.numbers import evenly, points


def training_curve_plots(curve: TrainingCurve) -> list[CurvePlot]:
    plots = [
        _per_epoch(
            "loss-curve",
            "Training loss per epoch",
            "The average loss on the training rows after each pass over them. It should fall "
            "quickly, then flatten; training stops once it stops improving.",
            "Training loss",
            curve.loss,
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
            )
        )
    return plots


def _per_epoch(
    plot_id: str, title: str, description: str, label: str, values: Sequence[float]
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
    )
