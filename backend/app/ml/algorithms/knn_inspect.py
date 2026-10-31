"""The KNN packs' evaluation views: a score per k, and the rows behind predictions (ADR 0025)."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from app.ml.base import Neighbour, NeighbourExample, NeighbourReport, TuningCurve
from app.ml.evaluation.numbers import round6

CANDIDATE_K = (1, 3, 5, 10, 20, 50)
#: Each k costs five fits and a prediction for every row; beyond this the view isn't worth the wait.
MAX_CURVE_ROWS = 20_000
CV_FOLDS = 5
#: Predictions to trace from each end: the model's biggest misses and its best.
EXAMPLES_PER_KIND = 3
#: Columns shown beside each neighbour; the rest are only counted.
MAX_COLUMNS = 5
#: Rows compared for the nearness figures; the split shuffled them, so the first few are random.
NEARNESS_QUERIES = 200
NEARNESS_ROWS = 2000


def k_curve(pipeline: Pipeline, X: pd.DataFrame, y: ArrayLike, score: str) -> TuningCurve | None:
    """`train.py.j2` runs the same loop, so the printed scores match this plot."""
    n_rows = len(X)
    if n_rows > MAX_CURVE_ROWS:
        return None
    chosen = pipeline.named_steps["model"].n_neighbors
    # Each fold trains on (CV_FOLDS - 1) / CV_FOLDS of the rows, which caps the usable k.
    largest = n_rows * (CV_FOLDS - 1) // CV_FOLDS
    candidates = sorted(k for k in {*CANDIDATE_K, chosen} if k <= largest)
    scores = [
        float(
            cross_val_score(
                clone(pipeline).set_params(model__n_neighbors=k), X, y, cv=CV_FOLDS
            ).mean()
        )
        for k in candidates
    ]
    return TuningCurve(
        name="n_neighbors",
        parameter="Neighbours (k)",
        values=candidates,
        scores=scores,
        score_label=score,
        chosen=chosen if chosen in candidates else None,
    )


def neighbour_report(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: NDArray[np.generic],
    errors: NDArray[np.float64],
) -> NeighbourReport:
    """`errors` ranks the test rows, worst first; each pack knows what an error is for its task."""
    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    order = np.argsort(-errors, kind="stable")
    misses = order[:EXAMPLES_PER_KIND]
    hits = [i for i in order[-EXAMPLES_PER_KIND:][::-1] if i not in misses]
    chosen = [*misses, *hits]

    distances, rows = model.kneighbors(preprocess.transform(X_test.iloc[chosen]))
    columns = _most_different(X_train, X_test.iloc[chosen], rows)
    examples = [
        NeighbourExample(
            kind="miss" if position < len(misses) else "hit",
            predicted=_text(predictions[row]),
            actual=_text(y_test.iloc[row]),
            values=[_text(X_test.iloc[row][c]) for c in columns],
            neighbours=[
                Neighbour(
                    distance=round6(float(d)),
                    share=round6(float(s)),
                    target=_text(y_train.iloc[n]),
                    values=[_text(X_train.iloc[n][c]) for c in columns],
                )
                for d, s, n in zip(
                    distances[position],
                    _shares(distances[position], model.weights),
                    rows[position],
                    strict=True,
                )
            ],
        )
        for position, row in enumerate(chosen)
    ]
    queries = preprocess.transform(X_test.head(NEARNESS_QUERIES))
    stored = preprocess.transform(X_train.head(NEARNESS_ROWS))
    return NeighbourReport(
        k=model.n_neighbors,
        stored_rows=model.n_samples_fit_,
        columns=columns,
        total_columns=X_train.shape[1],
        examples=examples,
        nearest_distance=round6(float(model.kneighbors(queries)[0][:, -1].mean())),
        typical_distance=round6(
            float(pairwise_distances(queries, stored, metric=model.metric).mean())
        ),
    )


def _shares(distances: NDArray[np.float64], weights: str) -> NDArray[np.float64]:
    if weights == "uniform":
        return np.full(len(distances), 1 / len(distances))
    exact = distances == 0
    say = exact.astype(float) if exact.any() else 1 / distances
    return say / say.sum()


def _most_different(
    X_train: pd.DataFrame, queries: pd.DataFrame, rows: NDArray[np.intp]
) -> list[str]:
    """Up to MAX_COLUMNS columns, widest gaps first: a number's in standard deviations, a
    category's as 0 (same) or 1 (different)."""
    gaps = {}
    for column in X_train.columns:
        values = X_train[column]
        stored, asked = values.to_numpy()[rows], queries[column].to_numpy()[:, None]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            gap = pd.Series(np.abs(stored - asked).ravel()).mean() / (values.std() or 1.0)
        else:
            gap = np.mean(stored != asked)
        gaps[column] = 0.0 if pd.isna(gap) else float(gap)  # all missing: nothing to compare
    return sorted(gaps, key=lambda c: -gaps[c])[:MAX_COLUMNS]


def _text(value: object) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float | np.floating):
        return f"{float(value):.4g}"
    return str(value)
