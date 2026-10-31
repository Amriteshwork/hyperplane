"""Pack-specific checks for KNN Classifier (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from app.ml.algorithms.knn_classifier import PACK
from app.ml.base import TaskType
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType
from app.ml.training import TrainingConfig, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "knn_classifier_scratch", Path("app/ml/algorithms/knn_classifier/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classes(n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_classes=n_classes,
        flip_y=0.1,
        random_state=0,
    )
    return X, np.array(["ant", "bee", "cat"])[y]  # string labels, as in a CSV


def circle() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    df["label"] = np.where(df["a"] ** 2 + df["b"] ** 2 > 1.4, "out", "in")
    return df


def config(**params: object) -> TrainingConfig:
    return TrainingConfig(
        target="label",
        task_type=TaskType.BINARY,
        preprocessing=PreprocessingConfig(numeric=("a", "b")),
        test_size=0.25,
        seed=0,
        params=PACK.params_model(**params).model_dump(mode="json"),
    )


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("k", [1, 4, 25])  # an even k has tied votes, which must break alike
def test_scratch_matches_scikit_learn_exactly(
    scratch: ModuleType, n_classes: int, weights: str, metric: str, k: int
) -> None:
    X, y = classes(n_classes)
    settings = {"n_neighbors": k, "weights": weights, "metric": metric}
    ours = scratch.KNNClassifierScratch(**settings).fit(X[:200], y[:200])
    theirs = KNeighborsClassifier(**settings, algorithm="brute").fit(X[:200], y[:200])

    np.testing.assert_array_equal(ours.classes_, theirs.classes_)
    np.testing.assert_allclose(ours.predict_proba(X[200:]), theirs.predict_proba(X[200:]))
    np.testing.assert_array_equal(ours.predict(X[200:]), theirs.predict(X[200:]))


def test_scratch_breaks_a_tied_vote_like_scikit_learn(scratch: ModuleType) -> None:
    X = np.array([[-1.0], [1.0]])
    y = np.array(["zebra", "ant"])
    query = np.array([[0.0]])
    ours = scratch.KNNClassifierScratch(n_neighbors=2).fit(X, y)
    theirs = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    assert ours.predict(query)[0] == theirs.predict(query)[0] == "ant"


def test_probabilities_are_shares_of_the_vote() -> None:
    X, y = classes(3)
    model = PACK.build_estimator(PACK.params_model(n_neighbors=5)).fit(X, y)
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(np.round(probabilities.ravel() * 5, 9)) <= {0, 1, 2, 3, 4, 5}


def test_k_of_one_memorises_the_training_rows() -> None:
    X, y = classes(2)
    model = PACK.build_estimator(PACK.params_model(n_neighbors=1)).fit(X, y)
    assert model.score(X, y) == 1.0


def test_it_draws_a_boundary_a_linear_model_cannot() -> None:
    df = circle()
    X, y = df[["a", "b"]], df["label"]
    knn = train(df, config(), PACK)
    accuracy = {m.label: m.value for m in knn.report.evaluation.metrics}["Accuracy"]
    assert accuracy > 0.9
    assert LogisticRegression().fit(X, y).score(X, y) < 0.8


def test_runs_report_probabilities_and_a_cross_validated_accuracy_per_k() -> None:
    result = train(circle(), config(), PACK)
    assert result.probabilities is not None
    plots = {p.id: p for p in result.report.evaluation.plots}
    assert {"confusion-matrix", "roc", "tuning-curve"} <= set(plots)
    assert plots["tuning-curve"].y_label == "Cross-validated accuracy"
    assert plots["tuning-curve"].marker is not None and plots["tuning-curve"].marker.x == 5
    assert result.report.coefficients == []


def test_the_biggest_misses_are_the_rows_the_right_class_lost_the_vote_on() -> None:
    df = circle()
    report = train(df, config(), PACK).report.neighbours
    assert report is not None
    misses = [e for e in report.examples if e.kind == "miss"]
    hits = [e for e in report.examples if e.kind == "hit"]
    assert any(e.predicted != e.actual for e in misses)
    assert all(e.predicted == e.actual for e in hits)
    for example in hits:  # the best are unanimous: all five said the actual class
        assert {n.target for n in example.neighbours} == {example.actual}
    for example in report.examples:
        votes = [n.target for n in example.neighbours]
        assert votes.count(example.predicted) == max(votes.count(v) for v in votes)


def test_a_class_never_seen_in_training_counts_as_a_miss() -> None:
    df = circle()
    result = train(df, config(), PACK)
    train_rows, test_rows = train_test_split(df, test_size=0.25, random_state=0)
    y_test = test_rows["label"].copy()
    y_test.iloc[:2] = "new"
    features = ["a", "b"]
    report = PACK.neighbours(
        result.pipeline, train_rows[features], train_rows["label"], test_rows[features], y_test
    )
    assert "new" in [e.actual for e in report.examples if e.kind == "miss"]


def profile(rows: int, features: int = 4) -> DatasetProfile:
    columns = [
        ColumnProfile(
            name=f"x{i}",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            missing=0,
            missing_pct=0.0,
            unique=10,
            unique_pct=1.0,
        )
        for i in range(features)
    ]
    return DatasetProfile(
        n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns, correlations=None
    )


def test_recommendation_prefers_few_features_and_refuses_regression() -> None:
    good = PACK.score(profile(5_000), TaskType.BINARY)
    wide = PACK.score(profile(5_000, features=60), TaskType.MULTICLASS)
    assert good.score > wide.score > 0
    assert any("Logistic Regression" in r for r in good.reasons)
    assert any("curse of dimensionality" in r for r in wide.reasons)
    assert any(r.startswith("When not to use") for r in good.reasons)

    refused = PACK.score(profile(300), TaskType.REGRESSION)
    assert refused.score == 0 and "KNN Regressor" in refused.reasons[0]
