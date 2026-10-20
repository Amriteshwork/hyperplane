"""Classification metrics (accuracy, precision, recall, F1, log-loss, ROC-AUC) and their plots.

Binary metrics treat the model's last class (`classes_[1]`, scikit-learn's positive class) as
positive; multiclass precision/recall/F1/ROC-AUC are macro averages: every class counts equally.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.ml.evaluation.models import (
    CurvePlot,
    CurveSeries,
    Evaluation,
    HeatmapPlot,
    Metric,
    Predictions,
)
from app.ml.evaluation.numbers import evenly, points


def evaluate_classification(predictions: Predictions) -> Evaluation:
    actual, predicted = np.asarray(predictions.y_true), np.asarray(predictions.y_pred)
    classes = list(predictions.classes) or sorted(set(actual) | set(predicted))
    proba = predictions.proba

    metrics = [_accuracy(actual, predicted), *_precision_recall_f1(actual, predicted, classes)]
    plots: list[Any] = [_confusion(actual, predicted, classes)]
    if proba is not None:
        metrics.append(_log_loss(actual, proba, classes))
        if _all_present(actual, classes):
            metrics.append(_roc_auc(actual, proba, classes))
            plots.append(_roc_curves(actual, proba, classes))
    return Evaluation(metrics=metrics, plots=plots)


def _accuracy(actual: NDArray[Any], predicted: NDArray[Any]) -> Metric:
    return Metric(
        id="accuracy",
        label="Accuracy",
        value=float(accuracy_score(actual, predicted)),
        description="Share of test rows given the right class. Misleading when one class "
        "dominates: always guessing the majority class can score high.",
        higher_is_better=True,
    )


def _precision_recall_f1(
    actual: NDArray[Any], predicted: NDArray[Any], classes: list[Any]
) -> list[Metric]:
    if len(classes) == 2:
        positive = classes[1]
        options: dict[str, Any] = {"average": "binary", "pos_label": positive}
        which = f"of the rows predicted {positive!s}"
        caught = f"of the actual {positive!s} rows"
        suffix = f" ({positive!s})"
    else:
        options = {"average": "macro", "labels": classes}
        which, caught, suffix = "per predicted class", "of each class", " (macro)"
    options["zero_division"] = 0
    return [
        Metric(
            id="precision",
            label=f"Precision{suffix}",
            value=float(precision_score(actual, predicted, **options)),
            description=f"Share {which} that were right: how much to trust a positive call."
            + _macro_note(classes),
            higher_is_better=True,
        ),
        Metric(
            id="recall",
            label=f"Recall{suffix}",
            value=float(recall_score(actual, predicted, **options)),
            description=f"Share {caught} that the model found: how few it misses."
            + _macro_note(classes),
            higher_is_better=True,
        ),
        Metric(
            id="f1",
            label=f"F1{suffix}",
            value=float(f1_score(actual, predicted, **options)),
            description="Harmonic mean of precision and recall: high only when both are."
            + _macro_note(classes),
            higher_is_better=True,
        ),
    ]


def _macro_note(classes: list[Any]) -> str:
    return "" if len(classes) == 2 else " Averaged over classes, each counting equally."


def _log_loss(actual: NDArray[Any], proba: NDArray[Any], classes: list[Any]) -> Metric:
    return Metric(
        id="log_loss",
        label="Log-loss",
        value=float(log_loss(actual, proba, labels=classes)),
        description="Average −log(probability given to the true class). Punishes confident "
        "wrong answers hardest; 0 is perfect.",
        higher_is_better=False,
    )


def _roc_auc(actual: NDArray[Any], proba: NDArray[Any], classes: list[Any]) -> Metric:
    if len(classes) == 2:
        value = roc_auc_score(actual == classes[1], proba[:, 1])
        label, note = "ROC-AUC", ""
    else:
        value = roc_auc_score(actual, proba, multi_class="ovr", average="macro", labels=classes)
        label, note = "ROC-AUC (macro)", " Each class against the rest, averaged."
    return Metric(
        id="roc_auc",
        label=label,
        value=float(value),
        description="Chance that a random positive row gets a higher score than a random "
        "negative one, over every threshold; 0.5 is guessing, 1 is perfect." + note,
        higher_is_better=True,
    )


def _all_present(actual: NDArray[Any], classes: list[Any]) -> bool:
    """ROC needs positives and negatives of every class it scores."""
    present = set(actual.tolist())
    return len(classes) >= 2 and all(c in present for c in classes)


def _confusion(actual: NDArray[Any], predicted: NDArray[Any], classes: list[Any]) -> HeatmapPlot:
    counts = confusion_matrix(actual, predicted, labels=classes)
    names = [str(c) for c in classes]
    return HeatmapPlot(
        id="confusion-matrix",
        title="Confusion matrix",
        description="Test rows by actual class (rows) and predicted class (columns). The "
        "diagonal is correct; every other cell is a kind of mistake.",
        x_label="Predicted",
        y_label="Actual",
        x_categories=names,
        y_categories=names,
        values=[[float(v) for v in row] for row in counts],
    )


def _roc_curves(actual: NDArray[Any], proba: NDArray[Any], classes: list[Any]) -> CurvePlot:
    # Binary: one curve for the positive class. Multiclass: each class against the rest.
    scored = [1] if len(classes) == 2 else range(len(classes))
    series = []
    for k in scored:
        fpr, tpr, _ = roc_curve(actual == classes[k], proba[:, k])
        kept = evenly(len(fpr))
        series.append(CurveSeries(name=str(classes[k]), points=points(fpr[kept], tpr[kept])))
    return CurvePlot(
        id="roc",
        title="ROC curve" if len(classes) == 2 else "ROC curves (one class vs the rest)",
        description="Each point is a threshold: the share of positives caught (true positive "
        "rate) against the share of negatives wrongly flagged (false positive rate). Curves "
        "hugging the top-left are better; the diagonal is guessing.",
        x_label="False positive rate",
        y_label="True positive rate",
        series=series,
        reference="diagonal",
    )
