"""Pack-specific checks for Logistic Regression (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from app.ml.algorithms.linear_regression import PACK as LINEAR_REGRESSION
from app.ml.algorithms.logistic_regression import PACK
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "logistic_regression_scratch", Path("app/ml/algorithms/logistic_regression/scratch.py")
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
    return X, y


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("C", [0.1, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType, n_classes: int, C: float, fit_intercept: bool
) -> None:
    X, y = classes(n_classes)
    ours = scratch.LogisticRegressionScratch(C, fit_intercept, tol=1e-10).fit(X, y)
    theirs = LogisticRegression(C=C, fit_intercept=fit_intercept, tol=1e-12, max_iter=100_000).fit(
        X, y
    )

    assert ours.n_iter_ < ours.max_iter  # converged, not stopped
    assert ours.coef_.shape == theirs.coef_.shape  # (1, p) for two classes, (K, p) for K
    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-6)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-6)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-6)
    np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))


def test_softmax_of_two_classes_is_the_sigmoid(scratch: ModuleType) -> None:
    z = np.linspace(-30, 30, 13)
    two = scratch.softmax(np.column_stack([np.zeros_like(z), z]))
    np.testing.assert_allclose(two[:, 1], scratch.sigmoid(z))
    assert np.all(np.isfinite(scratch.softmax(np.array([[1000.0, 0.0, -1000.0]]))))


def test_decision_boundary_is_where_the_probability_is_one_half(scratch: ModuleType) -> None:
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=1
    )
    model = scratch.LogisticRegressionScratch().fit(X, y)
    x1 = np.linspace(-2, 2, 5)
    on_line = np.column_stack([x1, scratch.decision_boundary(model, x1)])
    np.testing.assert_allclose(model.predict_proba(on_line)[:, 1], 0.5)


def test_class_coefficients_name_the_positive_class_or_every_class() -> None:
    X, y = classes(2)
    labels = np.array(["no", "yes"])[y]
    names = [f"x{i}" for i in range(X.shape[1])]
    binary = PACK.build_estimator(PACK.params_model()).fit(X, labels)
    assert list(PACK.class_coefficients(binary, names)) == ["yes"]  # scikit-learn's classes_[1]

    X3, y3 = classes(3)
    multi = PACK.build_estimator(PACK.params_model()).fit(X3, y3)
    table = PACK.class_coefficients(multi, names)
    assert list(table) == ["0", "1", "2"]
    assert table["1"]["x0"] == pytest.approx(multi.coef_[1, 0])


def test_class_weight_is_none_or_balanced() -> None:
    assert PACK.params_model(class_weight="balanced").model_dump()["class_weight"] == "balanced"
    assert PACK.params_model().model_dump()["class_weight"] is None
    with pytest.raises(ValidationError):
        PACK.params_model(class_weight="heavy")
    with pytest.raises(ValidationError):
        PACK.params_model(C=0)


def column(name: str, kind: SemanticType, unique: int = 10) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        semantic_type=kind,
        missing=0,
        missing_pct=0.0,
        unique=unique,
        unique_pct=1.0,
    )


def profile(rows: int, columns: list[ColumnProfile]) -> DatasetProfile:
    return DatasetProfile(
        n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns, correlations=None
    )


def numeric(n: int) -> list[ColumnProfile]:
    return [column(f"x{i}", SemanticType.NUMERIC) for i in range(n)]


def test_recommendation_refuses_regression() -> None:
    result = PACK.score(profile(100, numeric(3)), TaskType.REGRESSION)
    assert result.score == 0
    assert "Linear Regression" in result.reasons[0]


def test_recommendation_is_the_first_classifier_and_mentions_imbalance() -> None:
    binary = PACK.score(profile(1000, numeric(5)), TaskType.BINARY)
    multi = PACK.score(profile(1000, numeric(5)), TaskType.MULTICLASS)
    assert binary.score > multi.score > 0.5
    assert any("softmax" in r for r in multi.reasons)
    assert any("Class weights" in r for r in binary.reasons)
    assert any(r.startswith("When not to use") and "linear" in r for r in binary.reasons)


def test_recommendation_warns_about_few_rows_per_coefficient() -> None:
    scarce = PACK.score(profile(50, numeric(20)), TaskType.BINARY)
    assert scarce.score < PACK.score(profile(5000, numeric(20)), TaskType.BINARY).score
    assert any("rows per coefficient" in r for r in scarce.reasons)


def test_regression_packs_still_point_classification_here() -> None:
    result = LINEAR_REGRESSION.score(profile(100, numeric(3)), TaskType.BINARY)
    assert result.score == 0 and "Logistic Regression" in result.reasons[0]
