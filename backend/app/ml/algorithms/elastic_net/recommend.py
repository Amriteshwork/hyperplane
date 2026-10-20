"""Suitability rules: when to mix L1 and L2, and when one penalty alone is simpler."""

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
                    f"Not applicable: Elastic Net predicts a continuous number, but this target "
                    f"is {task.value}. Logistic Regression supports the same mixed penalty for "
                    "classes."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the target column; each category becomes one one-hot column.
    n_coefficients = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    value = 0.6
    reasons = [
        (
            "Lasso and Ridge in one: it drops unhelpful features like Lasso, but shares the "
            "weight between correlated ones like Ridge. It has two settings (α and the L1 "
            "ratio), best tuned together by cross-validation."
        )
    ]

    pairs = _correlated_pairs(profile)
    if pairs:
        value += 0.15
        reasons.append(
            f"{pairs} pair(s) of numeric columns are strongly correlated (|r| ≥ {_STRONG}). "
            "Lasso would keep one of each pair almost at random; the L2 part keeps them in or "
            "out together."
        )

    if n_coefficients >= _MANY:
        value += 0.1
        reasons.append(
            f"~{n_coefficients} coefficients: likely only some of them matter, and the L1 part "
            "can drop the rest."
        )

    rows_per_coefficient = profile.n_rows / max(n_coefficients, 1)
    if rows_per_coefficient < 20:
        value += 0.05
        reasons.append(
            f"Only ~{rows_per_coefficient:.0f} rows per coefficient: plain OLS would overfit; "
            "both penalties keep the model in check."
        )
    if n_coefficients > profile.n_rows:
        reasons.append(
            "More coefficients than rows: unlike Lasso, which keeps at most one feature per "
            "row, Elastic Net can keep as many as matter."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "When not to use: few, uncorrelated features (Ridge or Lasso does as well with one "
        "setting to tune instead of two), or the relationship is clearly curved (expand "
        "features first, or use a non-linear model)."
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
