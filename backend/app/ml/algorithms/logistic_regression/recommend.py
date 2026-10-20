"""Suitability rules: when a linear classifier is the right first model, and when it isn't."""

from itertools import combinations

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: |Pearson r| at which two columns are "strongly correlated".
_STRONG = 0.8


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: despite its name, Logistic Regression predicts classes, "
                    "but this target is a continuous number. Try Linear Regression."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the target column; each category becomes one one-hot column.
    n_coefficients = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    value = 0.8 if task is TaskType.BINARY else 0.75
    reasons = [
        (
            "The standard first classifier: fast, gives a probability for every class rather "
            "than just a label, and its coefficients say which features push towards which "
            "class."
        )
    ]
    if task is TaskType.MULTICLASS:
        reasons.append(
            "Several classes: a softmax turns one linear score per class into probabilities "
            "that add up to 1."
        )

    rows_per_coefficient = profile.n_rows / max(n_coefficients, 1)
    if rows_per_coefficient < 10:
        value -= 0.1
        reasons.append(
            f"Only ~{rows_per_coefficient:.0f} rows per coefficient: weights will be noisy. "
            "Keep C small (a stronger penalty) and compare a few values."
        )

    pairs = _correlated_pairs(profile)
    if pairs:
        reasons.append(
            f"{pairs} pair(s) of numeric columns are strongly correlated (|r| ≥ {_STRONG}): "
            "predictions are fine, but don't read their individual coefficients too literally."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "If one class is much rarer than the others, set Class weights to balanced and judge "
        "the model by recall, F1 or ROC-AUC rather than accuracy."
    )
    reasons.append(
        "When not to use: classes that a straight line (or plane) can't separate, such as "
        "rings or checkerboards, or strong interactions between features. Its decision "
        "boundary is linear; expand features first, or use a tree or kernel model."
    )
    return Suitability(score=round(min(max(value, 0.0), 1.0), 2), reasons=tuple(reasons))


def _correlated_pairs(profile: DatasetProfile) -> int:
    matrix = profile.correlations
    if matrix is None:
        return 0
    return sum(
        1
        for i, j in combinations(range(len(matrix.columns)), 2)
        if (r := matrix.values[i][j]) is not None and abs(r) >= _STRONG
    )
