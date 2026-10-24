"""Pack-specific checks for SGD Classifier (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.sgd_classifier import PACK
from app.ml.base import TaskType
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType
from app.ml.training import TrainingConfig, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sgd_classifier_scratch", Path("app/ml/algorithms/sgd_classifier/scratch.py")
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
        flip_y=0.1,  # label noise: no perfect separation, so the minimum is well-conditioned
        random_state=0,
    )
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("n_classes", [2, 3])
def test_scratch_log_loss_reaches_logistic_regressions_minimum(
    scratch: ModuleType, n_classes: int
) -> None:
    # Log-loss + L2 per row is Logistic Regression's objective with C = 1/(α·n), one-vs-rest
    # for 3+ classes. 200 epochs of the 1/t schedule land within the documented 5e-3.
    alpha = 0.01
    X, y = classes(n_classes)
    ours = scratch.SGDClassifierScratch(alpha=alpha, max_iter=200, tol=None).fit(X, y)
    exact = LogisticRegression(C=1 / (alpha * len(y)), tol=1e-10, max_iter=10_000)
    theirs = (exact if n_classes == 2 else OneVsRestClassifier(exact)).fit(X, y)

    assert ours.coef_.shape == (1 if n_classes == 2 else n_classes, X.shape[1])
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=5e-3)
    np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))


def test_scratch_hinge_has_no_probabilities_and_rivals_scikit_learn(scratch: ModuleType) -> None:
    X, y = classes(2)
    ours = scratch.SGDClassifierScratch(loss="hinge").fit(X, y)
    theirs = SGDClassifier(loss="hinge", random_state=0).fit(X, y)
    assert abs((ours.predict(X) == y).mean() - theirs.score(X, y)) < 0.1
    with pytest.raises(AttributeError):
        ours.predict_proba(X)


def test_class_coefficients_name_the_positive_class_or_every_class() -> None:
    X, y = classes(2)
    names = [f"x{i}" for i in range(X.shape[1])]
    binary = PACK.build_estimator(PACK.params_model()).fit(X, np.array(["no", "yes"])[y])
    assert list(PACK.class_coefficients(binary, names)) == ["yes"]  # scikit-learn's classes_[1]

    X3, y3 = classes(3)
    multi = PACK.build_estimator(PACK.params_model()).fit(X3, y3)
    table = PACK.class_coefficients(multi, names)
    assert list(table) == ["0", "1", "2"]
    assert table["1"]["x0"] == pytest.approx(multi.coef_[1, 0])


def test_defaults_are_scikit_learns_but_log_loss_and_a_fixed_seed() -> None:
    ours = PACK.params_model().model_dump()
    theirs = SGDClassifier().get_params()
    changed = {k for k in ours if ours[k] != theirs[k]}
    assert changed == {"loss", "random_state"}
    assert ours["loss"] == "log_loss"  # probabilities, so log-loss and ROC-AUC are reported


def test_params_reject_values_scikit_learn_would() -> None:
    with pytest.raises(ValidationError):
        PACK.params_model(alpha=0)
    with pytest.raises(ValidationError):
        PACK.params_model(loss="squared_error")  # a regression loss
    with pytest.raises(ValidationError):
        PACK.params_model(class_weight="heavy")


def test_hinge_runs_report_label_metrics_only() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    df["label"] = np.where(df["a"] - df["b"] > 0, "up", "down")
    config = TrainingConfig(
        target="label",
        task_type=TaskType.BINARY,
        preprocessing=PreprocessingConfig(numeric=("a", "b")),
        test_size=0.25,
        seed=0,
        params=PACK.params_model(loss="hinge").model_dump(mode="json"),
    )
    result = train(df, config, PACK)
    assert result.probabilities is None  # no predict_proba for the hinge loss
    assert [m.id for m in result.report.evaluation.metrics] == [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    assert [p.kind for p in result.report.evaluation.plots] == ["heatmap"]  # no ROC curve


def profile(rows: int) -> DatasetProfile:
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
        for i in range(4)
    ]
    return DatasetProfile(
        n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns, correlations=None
    )


def test_recommendation_prefers_large_data_and_refuses_regression() -> None:
    small = PACK.score(profile(500), TaskType.BINARY)
    large = PACK.score(profile(500_000), TaskType.BINARY)
    assert large.score > small.score > 0
    assert any("Logistic Regression" in r for r in small.reasons)
    assert any("one-vs-rest" in r for r in PACK.score(profile(500), TaskType.MULTICLASS).reasons)
    assert any(r.startswith("When not to use") for r in large.reasons)

    refused = PACK.score(profile(500), TaskType.REGRESSION)
    assert refused.score == 0 and "SGD Regressor" in refused.reasons[0]
