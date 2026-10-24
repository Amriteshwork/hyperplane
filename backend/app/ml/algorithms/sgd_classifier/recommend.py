"""Suitability rules: SGD pays off on large datasets; on small ones Logistic Regression is simpler."""

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile

#: From this many rows SGD's one-row-at-a-time updates start to beat exact solvers.
_LARGE = 100_000
#: Below this many rows an epoch is tiny, so SGD's answer is noisy and L-BFGS is instant.
_SMALL = 10_000


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: SGD Classifier predicts classes, but this target is a "
                    "continuous number. Try SGD Regressor, the same training method for numbers."
                ),
            ),
        )

    reasons = [
        (
            "A linear classifier trained by stochastic gradient descent: one small update per "
            "row, so it scales to datasets too large for Logistic Regression's solver. Its loss "
            "picks the model: log_loss is logistic regression, hinge a linear SVM."
        )
    ]
    if profile.n_rows >= _LARGE:
        value = 0.8
        reasons.append(
            f"{profile.n_rows:,} rows: exact solvers slow down here, while SGD reaches a good "
            "fit within a few passes over the data."
        )
    elif profile.n_rows >= _SMALL:
        value = 0.6
        reasons.append(
            f"{profile.n_rows:,} rows: SGD trains quickly, though Logistic Regression would "
            "still fit this in about the same time."
        )
    else:
        value = 0.4
        reasons.append(
            f"Only {profile.n_rows:,} rows: Logistic Regression finds the exact minimum "
            "instantly, with no learning rate to tune. Use SGD here to learn how it behaves."
        )
    if task is TaskType.MULTICLASS:
        reasons.append(
            "Several classes: SGD trains one yes/no model per class (one-vs-rest) and picks "
            "the most confident."
        )

    missing = [c.name for c in profile.columns if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "If one class is much rarer than the others, set Class weights to balanced and judge "
        "the model by recall, F1 or ROC-AUC rather than accuracy."
    )
    reasons.append(
        "When not to use: small datasets (Logistic Regression is exact and deterministic), "
        "classes a straight boundary can't separate (use a tree or kernel model), or when you "
        "need identical weights on every run."
    )
    return Suitability(score=value, reasons=tuple(reasons))
