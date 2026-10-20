"""Suitability rules: when shrinking OLS's coefficients pays off, and when it can't help."""

from itertools import combinations

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: |Pearson r| at which two columns are "strongly correlated".
_STRONG = 0.8


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    f"Not applicable: Ridge predicts a continuous number, but this target is "
                    f"{task.value}. Logistic Regression has the same L2 penalty for classes."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the target column; each category becomes one one-hot column.
    n_coefficients = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    value = 0.7
    reasons = [
        (
            "A safer Linear Regression: the same weighted sum, plus a penalty on large "
            "coefficients, so it overfits less. Tune α by comparing test scores (or with "
            "cross-validation)."
        )
    ]

    rows_per_coefficient = profile.n_rows / max(n_coefficients, 1)
    if rows_per_coefficient < 20:
        value += 0.15
        reasons.append(
            f"Only ~{rows_per_coefficient:.0f} rows per coefficient: plain OLS would be "
            "unstable here, which is exactly what the penalty fixes."
        )

    pairs = _correlated_pairs(profile)
    if pairs:
        value += 0.1
        reasons.append(
            f"{pairs} pair(s) of numeric columns are strongly correlated (|r| ≥ {_STRONG}). "
            "Between features, that makes OLS coefficients swing; Ridge shares the weight "
            "between them instead."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "When not to use: you want a short list of the features that matter (Ridge keeps "
        "every feature; use Lasso), or the relationship is clearly curved (expand features "
        "first, or use a non-linear model)."
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
