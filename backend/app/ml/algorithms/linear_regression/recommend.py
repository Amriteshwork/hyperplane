"""Suitability rules: when OLS is a good first model, and when it isn't."""

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    f"Not applicable: Linear Regression predicts a continuous number, but this "
                    f"target is {task.value}. Try Logistic Regression for classes."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    numeric = [c for c in usable if c.semantic_type is not SemanticType.CATEGORICAL]
    # The profile includes the target column, so features ≈ usable − 1.
    n_features = max(len(usable) - 1, 1)
    value = 0.75
    reasons = [
        (
            "Good baseline: the target is continuous, and OLS is fast, has no settings to tune, "
            "and its coefficients are easy to read."
        )
    ]

    if len(numeric) >= 0.8 * len(usable):
        value += 0.1
        reasons.append("Features are mostly numeric, which linear models use directly.")
    else:
        reasons.append(
            "Several categorical features will be one-hot encoded; with many categories the "
            "model gets many coefficients."
        )

    rows_per_feature = profile.n_rows / n_features
    if rows_per_feature < 20:
        value -= 0.25
        reasons.append(
            f"Only ~{rows_per_feature:.0f} rows per feature: coefficients will be unstable. "
            "Ridge (L2 regularisation) usually generalises better here."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(
            f"{len(missing)} column(s) have missing values; they will be imputed first, since "
            "OLS cannot handle gaps."
        )

    reasons.append(
        "When not to use: strongly non-linear relationships or interacting features. Check "
        "the residual plot; curved patterns mean a straight line is the wrong shape."
    )
    return Suitability(score=round(min(max(value, 0.0), 1.0), 2), reasons=tuple(reasons))
