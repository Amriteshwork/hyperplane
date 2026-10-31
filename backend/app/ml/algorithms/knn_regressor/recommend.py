"""Suitability rules: KNN fits curved relationships when rows are plentiful and features few."""

from app.ml.algorithms.knn_recommend import data_fit
from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: KNN Regressor predicts numbers, but this target is a "
                    "class. Try KNN Classifier, the same search with a vote at the end."
                ),
            ),
        )

    value, reasons = data_fit(profile)
    reasons.insert(
        0,
        "No formula to fit: it predicts the average target of the most similar training rows, "
        "so it can follow curves and interactions without you designing any features.",
    )
    reasons.append(
        "When not to use: when you must predict beyond the range of the training targets (an "
        "average of neighbours can never leave it), when predictions must be fast, when the "
        "data is high-dimensional, or when you need coefficients to explain the effects."
    )
    return Suitability(score=round(min(max(value, 0.0), 1.0), 2), reasons=tuple(reasons))
