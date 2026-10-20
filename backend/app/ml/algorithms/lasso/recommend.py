"""Suitability rules: when only some features should matter, and when Lasso picks badly."""

from itertools import combinations

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: |Pearson r| at which two columns are "strongly correlated".
_STRONG = 0.8
#: From this many coefficients, a shorter list of the ones that matter is worth having.
_MANY = 10


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    f"Not applicable: Lasso predicts a continuous number, but this target is "
                    f"{task.value}. Logistic Regression has the same L1 penalty for classes."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the target column; each category becomes one one-hot column.
    n_coefficients = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    value = 0.65
    reasons = [
        (
            "A Linear Regression that also selects features: its penalty sets the coefficients "
            "of unhelpful features exactly to 0, leaving a shorter model that is easier to "
            "read. Tune α by comparing test scores (or with cross-validation)."
        )
    ]

    if n_coefficients >= _MANY:
        value += 0.15
        reasons.append(
            f"~{n_coefficients} coefficients: likely only some of them matter. Lasso finds "
            "which, and drops the rest."
        )

    rows_per_coefficient = profile.n_rows / max(n_coefficients, 1)
    if rows_per_coefficient < 20:
        value += 0.1
        reasons.append(
            f"Only ~{rows_per_coefficient:.0f} rows per coefficient: plain OLS would overfit, "
            "and the penalty keeps the model small."
        )
    if n_coefficients > profile.n_rows:
        reasons.append(
            f"More coefficients than rows: Lasso keeps at most {profile.n_rows} features, one "
            "per row, even if more matter."
        )

    pairs = _correlated_pairs(profile)
    if pairs:
        value -= 0.1
        reasons.append(
            f"{pairs} pair(s) of numeric columns are strongly correlated (|r| ≥ {_STRONG}). "
            "Lasso tends to keep one of each pair and drop the other, almost at random; Ridge "
            "or Elastic Net share the weight instead."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "When not to use: most features genuinely matter (Ridge shrinks without dropping "
        "any), features come in correlated groups you want kept together (Elastic Net), or "
        "the relationship is clearly curved (expand features first, or use a non-linear model)."
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
