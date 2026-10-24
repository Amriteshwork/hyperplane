"""Suitability rules: an MLP needs thousands of rows to beat simpler models on tabular data."""

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile

#: From this many rows a network has enough examples to learn non-linear features reliably.
_LARGE = 10_000
#: Below this many rows a 100-neuron network mostly memorises noise.
_SMALL = 1_000


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: MLP Classifier predicts classes, but this target is a "
                    "continuous number. Try MLP Regressor, the same network with a numeric output."
                ),
            ),
        )

    reasons = [
        (
            "A neural network: layers of neurons that each bend a weighted sum, so its "
            "decision boundary can curve around the classes where Logistic Regression's must "
            "stay straight."
        )
    ]
    if profile.n_rows >= _LARGE:
        value = 0.7
        reasons.append(
            f"{profile.n_rows:,} rows: enough for the network to learn non-linear features "
            "without memorising the training set."
        )
    elif profile.n_rows >= _SMALL:
        value = 0.5
        reasons.append(
            f"{profile.n_rows:,} rows: workable, but watch the gap between training and test "
            "accuracy and raise α if it grows."
        )
    else:
        value = 0.3
        reasons.append(
            f"Only {profile.n_rows:,} rows: a network has far more weights than that and "
            "tends to fit noise. Logistic Regression is likely as good; use an MLP here to "
            "learn how it behaves."
        )
    if task is TaskType.MULTICLASS:
        reasons.append(
            "Several classes: one output neuron per class, turned into probabilities by a "
            "softmax, all trained together."
        )

    missing = [c.name for c in profile.columns if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "When not to use: small datasets, when you need to explain each feature's effect (an "
        "MLP has no coefficients to read), or when you need the same answer on every seed. On "
        "tabular data, gradient-boosted trees are often more accurate for less tuning."
    )
    return Suitability(score=value, reasons=tuple(reasons))
