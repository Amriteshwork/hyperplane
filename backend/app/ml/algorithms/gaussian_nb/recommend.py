"""Suitability rules: Gaussian NB suits few rows and roughly bell-shaped, unrelated features."""

from itertools import combinations

from app.ml.base import Suitability, TaskType
from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: |Pearson r| at which two columns start to count the same evidence twice.
_STRONG = 0.8
#: |skew| past which a single bell curve fits a column badly.
_SKEWED = 2.0
#: Fewer rows than this, and models with more to estimate are likely to overfit.
_FEW_ROWS = 1_000


def score(profile: DatasetProfile, task: TaskType) -> Suitability:
    if task is TaskType.REGRESSION:
        return Suitability(
            score=0.0,
            reasons=(
                (
                    "Not applicable: Gaussian Naive Bayes predicts classes, but this target is a "
                    "continuous number. Try Linear Regression."
                ),
            ),
        )

    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    value = 0.6
    reasons = [
        (
            "A fast baseline with no training loop: it estimates one mean and one variance per "
            "class and feature, then combines them with Bayes' theorem."
        )
    ]

    if profile.n_rows < _FEW_ROWS:
        value += 0.1
        reasons.append(
            f"{profile.n_rows:,} rows: each feature is estimated on its own, so it needs few "
            "rows, and it can beat Logistic Regression while data is scarce."
        )

    pairs = _correlated_pairs(profile)
    if pairs:
        value -= 0.1
        reasons.append(
            f"{pairs} pair(s) of numeric columns are strongly correlated (|r| ≥ {_STRONG}): it "
            "counts their evidence twice, so its probabilities will be overconfident even if its "
            "labels are good."
        )

    skewed = [
        c
        for c in usable
        if c.numeric is not None and c.numeric.skew is not None and abs(c.numeric.skew) > _SKEWED
    ]
    if skewed:
        value -= 0.1
        reasons.append(
            f"{len(skewed)} column(s) are heavily skewed, so one bell curve fits them badly. A "
            "log transform before uploading may help."
        )

    if categorical:
        value -= 0.1
        reasons.append(
            f"{len(categorical)} categorical column(s) become one-hot 0s and 1s, which are not "
            "bell-shaped. A category never seen in a class can dominate the sum, so check the "
            "smoothing curve in the results. Categorical or Bernoulli Naive Bayes fits them "
            "better."
        )

    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")

    reasons.append(
        "When not to use: when you need trustworthy probabilities (they get pushed towards 0 and "
        "1), when features depend on each other in ways that matter (it can't learn XOR), or "
        "with plenty of data, where Logistic Regression or trees usually score higher."
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
