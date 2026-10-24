"""Suitability rules: SGD pays off on large datasets; on small ones an exact solver is simpler."""

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile

#: From this many rows SGD's one-row-at-a-time updates start to beat exact solvers.
_LARGE = 100_000
#: Below this many rows an epoch is tiny, so SGD's answer is noisy and an exact one is instant.
_SMALL = 10_000


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is not TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    f"Not applicable: SGD Regressor predicts a continuous number, but this "
                    f"target is {task.value}. Try SGD Classifier, the same training method "
                    "for classes."
                ),
            ),
        )

    reasons = [
        (
            "A linear model trained by stochastic gradient descent: one small update per row, "
            "so it scales to datasets too large for exact solvers and teaches the optimiser "
            "behind neural networks."
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
            f"{profile.n_rows:,} rows: SGD trains quickly, though Ridge would still solve this "
            "exactly in about the same time."
        )
    else:
        value = 0.4
        reasons.append(
            f"Only {profile.n_rows:,} rows: Linear Regression or Ridge give the exact answer "
            "instantly, with no learning rate to tune. Use SGD here to learn how it behaves."
        )

    missing = [c.name for c in profile.columns if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "Features are standardised before training: SGD's step size is shared by every "
        "feature, so unscaled ones make it crawl or diverge."
    )
    reasons.append(
        "When not to use: small datasets (an exact solver is simpler and deterministic), "
        "clearly curved relationships (expand features first, or use a non-linear model), or "
        "when you need identical coefficients on every run."
    )
    return Suitability(score=value, reasons=tuple(reasons))
