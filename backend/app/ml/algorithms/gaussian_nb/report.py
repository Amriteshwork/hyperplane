"""Gaussian NB's evaluation views: a score per smoothing, the bell curves, and the sums (ADR 0026)."""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from app.ml.base import (
    CorrelatedPair,
    Evidence,
    FeatureGaussians,
    GaussianExample,
    GaussianReport,
    TuningCurve,
)
from app.ml.evaluation.numbers import round6

CANDIDATE_SMOOTHING = (1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 1.0)
CV_FOLDS = 5
#: Predictions to explain from each end: the model's biggest misses and its best.
EXAMPLES_PER_KIND = 3
#: Features drawn as bell curves, and listed in a prediction's sum; the rest are summed together.
MAX_FEATURES = 8
MAX_EVIDENCE = 5
#: Columns compared for correlation, so a very wide dataset doesn't need a huge matrix.
MAX_CORRELATED = 100


def smoothing_curve(pipeline: Pipeline, X: pd.DataFrame, y: ArrayLike) -> TuningCurve:
    """`train.py.j2` runs the same loop, so the printed scores match this plot."""
    chosen = float(pipeline.named_steps["model"].var_smoothing)
    candidates = sorted({*CANDIDATE_SMOOTHING, chosen})
    scores = [
        float(
            cross_val_score(
                clone(pipeline).set_params(model__var_smoothing=value), X, y, cv=CV_FOLDS
            ).mean()
        )
        for value in candidates
    ]
    return TuningCurve(
        name="var_smoothing",
        parameter="Variance smoothing",
        values=candidates,
        scores=scores,
        score_label="Cross-validated accuracy",
        chosen=chosen,
        log_scale=True,
    )


def gaussian_report(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> GaussianReport:
    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    names = [str(n) for n in preprocess.get_feature_names_out()]
    n_numeric = sum(n.startswith("numeric__") for n in names)
    scale, shift = _own_units(preprocess, len(names), n_numeric)
    stds = np.sqrt(model.var_)

    separation = model.theta_.var(axis=0) / model.var_.mean(axis=0)
    features = [
        FeatureGaussians(
            feature=names[column],
            means=[round6(v) for v in model.theta_[:, column] * scale[column] + shift[column]],
            stds=[round6(v) for v in stds[:, column] * scale[column]],
            separation=round6(separation[column]),
        )
        for column in np.argsort(-separation, kind="stable")[:MAX_FEATURES]
    ]

    probabilities = pipeline.predict_proba(X_test)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    # How little probability the right class got; a class unseen in training got none.
    actual = pd.Index(model.classes_).get_indexer(y_test)
    got = probabilities[np.arange(len(actual)), actual]
    order = np.argsort(np.where(actual >= 0, got, 0.0), kind="stable")
    misses = order[:EXAMPLES_PER_KIND]
    hits = [i for i in order[-EXAMPLES_PER_KIND:][::-1] if i not in misses]

    rows = preprocess.transform(X_test.iloc[[*misses, *hits]])
    examples = [
        _example(
            "miss" if position < len(misses) else "hit",
            row,
            probabilities[index],
            actual[index],
            model,
            names,
            scale,
            shift,
        )
        for position, (row, index) in enumerate(zip(rows, [*misses, *hits], strict=True))
    ]
    return GaussianReport(
        classes=[str(c) for c in model.classes_],
        priors=[round6(p) for p in model.class_prior_],
        features=features,
        total_features=len(names),
        examples=examples,
        mean_confidence=round6(probabilities.max(axis=1).mean()),
        accuracy=round6(accuracy_score(y_test, predictions)),
        correlated=_most_correlated(preprocess, model, X_train, y_train, names, n_numeric),
    )


def _own_units(
    preprocess: ColumnTransformer, n_columns: int, n_numeric: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """How to undo standardising: the numeric columns come first, and one-hot columns stay as is."""
    scale, shift = np.ones(n_columns), np.zeros(n_columns)
    if n_numeric:
        scaler = preprocess.named_transformers_["numeric"].named_steps["scale"]
        scale[:n_numeric], shift[:n_numeric] = scaler.scale_, scaler.mean_
    return scale, shift


def _example(
    kind: Literal["miss", "hit"],
    row: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    actual: int,
    model: GaussianNB,
    names: list[str],
    scale: NDArray[np.float64],
    shift: NDArray[np.float64],
) -> GaussianExample:
    classes, priors = model.classes_, model.class_prior_
    theta, var = model.theta_, model.var_
    ranked = np.argsort(-probabilities, kind="stable")
    predicted = int(ranked[0])
    # Compare with the right answer when the model missed it, else with its closest alternative.
    rival = actual if actual >= 0 and actual != predicted else int(ranked[1])
    log_bell = -0.5 * np.log(2 * np.pi * var) - (row - theta) ** 2 / (2 * var)
    gap = log_bell[predicted] - log_bell[rival]
    order = np.argsort(-np.abs(gap), kind="stable")

    def entry(feature: str, value: str, first: float, second: float) -> Evidence:
        return Evidence(feature=feature, value=value, predicted=round6(first), rival=round6(second))

    shown, rest = order[:MAX_EVIDENCE], order[MAX_EVIDENCE:]
    return GaussianExample(
        kind=kind,
        predicted=str(classes[predicted]),
        actual=str(classes[actual]) if actual >= 0 else "not seen in training",
        rival=str(classes[rival]),
        confidence=round6(probabilities[predicted]),
        prior=entry("Prior", "", float(np.log(priors[predicted])), float(np.log(priors[rival]))),
        evidence=[
            entry(
                names[c],
                f"{row[c] * scale[c] + shift[c]:.4g}",
                log_bell[predicted, c],
                log_bell[rival, c],
            )
            for c in shown
        ],
        others=entry(
            f"{len(rest)} other columns",
            "",
            log_bell[predicted, rest].sum(),
            log_bell[rival, rest].sum(),
        )
        if len(rest)
        else None,
    )


def _most_correlated(
    preprocess: ColumnTransformer,
    model: GaussianNB,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    names: list[str],
    n_numeric: int,
) -> CorrelatedPair | None:
    """Numeric columns only: the one-hot columns of one category always move together."""
    columns = min(n_numeric, MAX_CORRELATED)
    if columns < 2:
        return None
    rows = preprocess.transform(X_train)[:, :columns]
    classes, theta = model.classes_, model.theta_
    residuals = rows - theta[np.searchsorted(classes, y_train), :columns]
    varying = np.flatnonzero(residuals.std(axis=0) > 0)
    if len(varying) < 2:
        return None
    correlations = np.corrcoef(residuals[:, varying], rowvar=False)
    np.fill_diagonal(correlations, 0.0)
    i, j = np.unravel_index(np.abs(correlations).argmax(), correlations.shape)
    return CorrelatedPair(
        a=names[varying[i]], b=names[varying[j]], correlation=round6(correlations[i, j])
    )
