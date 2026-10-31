"""Suitability rules: KNN draws curved boundaries when rows are plentiful and features few."""

from app.ml.algorithms.knn_recommend import data_fit
from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: KNN Classifier predicts classes, but this target is a "
                    "continuous number. Try KNN Regressor, the same search with an average at "
                    "the end."
                ),
            ),
        )

    value, reasons = data_fit(profile)
    reasons.insert(
        0,
        "No formula to fit: the most similar training rows vote, so the decision boundary can "
        "take any shape where Logistic Regression's must stay straight.",
    )
    reasons.append(
        "If one class is much rarer, it gets outvoted in most neighbourhoods: try distance "
        "weights and judge the model by recall or F1 rather than accuracy."
    )
    reasons.append(
        "When not to use: when predictions must be fast, when the data is high-dimensional, "
        "when many features are irrelevant (each adds noise to every distance), or when you "
        "need calibrated probabilities (these are just shares of k votes)."
    )
    return Suitability(score=round(min(max(value, 0.0), 1.0), 2), reasons=tuple(reasons))
