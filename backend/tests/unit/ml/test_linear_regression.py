"""Pack-specific checks for Linear Regression (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.linear_regression import PACK
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lr_scratch", Path("app/ml/algorithms/linear_regression/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=0)
    return X, y


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], fit_intercept: bool
) -> None:
    X, y = data
    ours = scratch.LinearRegressionScratch(fit_intercept=fit_intercept).fit(X, y)
    theirs = LinearRegression(fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)
    assert scratch.r2_score(y, ours.predict(X)) == pytest.approx(theirs.score(X, y), abs=1e-10)


def test_gradient_descent_converges_to_the_closed_form(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    X = StandardScaler().fit_transform(X)
    w, b = scratch.fit_gradient_descent(X, y)
    exact = LinearRegression().fit(X, y)
    np.testing.assert_allclose(w, exact.coef_, rtol=1e-4)
    assert b == pytest.approx(exact.intercept_, rel=1e-4)


def column(name: str, kind: SemanticType, missing: int = 0) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        semantic_type=kind,
        missing=missing,
        missing_pct=0.0,
        unique=10,
        unique_pct=1.0,
    )


def profile(rows: int, columns: list[ColumnProfile]) -> DatasetProfile:
    return DatasetProfile(n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns)


def test_recommendation_refuses_classification() -> None:
    result = PACK.score(profile(100, [column("a", SemanticType.NUMERIC)]), TaskType.BINARY)
    assert result.score == 0
    assert "Logistic Regression" in result.reasons[0]


def test_recommendation_prefers_many_rows_of_numeric_features() -> None:
    numeric = [column(c, SemanticType.NUMERIC) for c in "abcd"]
    good = PACK.score(profile(1000, numeric), TaskType.REGRESSION)
    few_rows = PACK.score(profile(30, numeric), TaskType.REGRESSION)

    assert good.score > few_rows.score
    assert any("Ridge" in r for r in few_rows.reasons)
    assert any(r.startswith("When not to use") for r in good.reasons)
