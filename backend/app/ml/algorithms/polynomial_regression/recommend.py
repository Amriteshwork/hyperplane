"""Suitability rules: when bending a linear model is worth it, and when the terms explode."""

from math import comb

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: The default degree; the estimate shows what the learner gets without changing anything.
_DEGREE = 2


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    f"Not applicable: Polynomial Regression predicts a continuous number, but "
                    f"this target is {task.value}. Try Logistic Regression for classes."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the (numeric) target; each category becomes one one-hot column.
    n_inputs = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    n_terms = comb(max(n_inputs, 1) + _DEGREE, _DEGREE) - 1
    value = 0.6
    reasons = [
        (
            "Worth trying when a straight line underfits: it adds squares and products of the "
            "features, so the same least-squares fit can follow curves and interactions."
        ),
        (
            f"With degree {_DEGREE}, ~{n_inputs} input column(s) become ~{n_terms} terms, "
            "each with its own coefficient."
        ),
    ]

    if n_inputs <= 10:
        value += 0.1
        reasons.append("Few input columns, so the expanded model stays small and readable.")
    else:
        value -= 0.15
        reasons.append(
            "Many input columns: the number of terms grows quadratically with degree 2 (and "
            "faster beyond), so coefficients become hard to read and slow to fit."
        )

    rows_per_term = profile.n_rows / n_terms
    if rows_per_term < 20:
        value -= 0.25
        reasons.append(
            f"Only ~{rows_per_term:.0f} rows per term: expect overfitting. Lower the degree, "
            "or use Ridge to keep the coefficients small."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(
            f"{len(missing)} column(s) have missing values; they will be imputed before the "
            "features are expanded."
        )

    reasons.append(
        "When not to use: many features, high degrees, or predictions outside the training "
        "range. Polynomials shoot off to ±∞ beyond the data they were fitted on."
    )
    return Suitability(score=round(min(max(value, 0.0), 1.0), 2), reasons=tuple(reasons))
