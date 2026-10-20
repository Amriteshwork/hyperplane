"""Pack-specific checks for Ridge (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from app.ml.algorithms.linear_regression import PACK as LINEAR_REGRESSION
from app.ml.algorithms.ridge import PACK
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, CorrelationMatrix, SemanticType


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ridge_scratch", Path("app/ml/algorithms/ridge/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=6, noise=10.0, random_state=0)
    return X, y


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], alpha: float, fit_intercept: bool
) -> None:
    X, y = data
    ours = scratch.RidgeScratch(alpha, fit_intercept=fit_intercept).fit(X, y)
    theirs = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)


def test_scratch_solves_when_features_are_collinear(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    X = np.column_stack([x, x])  # XᵀX is singular: OLS has no unique solution
    model = scratch.RidgeScratch(1.0).fit(X, 2 * x)
    assert model.coef_[0] == pytest.approx(model.coef_[1])
    np.testing.assert_allclose(model.coef_, Ridge(alpha=1.0).fit(X, 2 * x).coef_, rtol=1e-8)


def test_path_shrinks_towards_ols_and_zero(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    path = scratch.ridge_path(X, y, np.logspace(-6, 6, 13))
    norms = np.linalg.norm(path, axis=1)

    assert np.all(np.diff(norms) < 0)  # every step up in α shrinks the coefficients
    np.testing.assert_allclose(path[0], LinearRegression().fit(X, y).coef_, rtol=1e-4)
    assert norms[-1] < 0.01 * norms[0]


def test_cross_validation_prefers_a_penalty_when_rows_are_scarce(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40))
    y = X[:, :3] @ np.array([3.0, -2.0, 1.0]) + rng.normal(scale=2.0, size=60)
    alphas = np.array([1e-6, 1.0, 10.0, 100.0, 1e5])
    best, scores = scratch.choose_alpha(X, y, alphas)

    assert best == alphas[np.argmin(scores)]
    assert 1e-6 < best < 1e5  # neither no penalty nor a crushing one
    assert scores.shape == alphas.shape


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


def profile(
    rows: int, columns: list[ColumnProfile], correlations: CorrelationMatrix | None = None
) -> DatasetProfile:
    return DatasetProfile(
        n_rows=rows,
        n_columns=len(columns),
        duplicate_rows=0,
        columns=columns,
        correlations=correlations,
    )


def test_recommendation_refuses_classification() -> None:
    result = PACK.score(profile(100, [column("a", SemanticType.NUMERIC)]), TaskType.BINARY)
    assert result.score == 0
    assert "Logistic Regression" in result.reasons[0]


def test_recommendation_favours_few_rows_and_correlated_features() -> None:
    numeric = [column(c, SemanticType.NUMERIC) for c in "abcd"]
    correlated = CorrelationMatrix(
        columns=["a", "b"],
        values=[[1.0, 0.95], [0.95, 1.0]],
    )
    plain = PACK.score(profile(10_000, numeric), TaskType.REGRESSION)
    scarce = PACK.score(profile(30, numeric), TaskType.REGRESSION)
    collinear = PACK.score(profile(10_000, numeric, correlated), TaskType.REGRESSION)

    assert scarce.score > plain.score
    assert collinear.score > plain.score
    assert any("strongly correlated" in r for r in collinear.reasons)
    assert any(r.startswith("When not to use") and "Lasso" in r for r in plain.reasons)


def test_ridge_outranks_linear_regression_when_rows_are_scarce() -> None:
    numeric = [column(c, SemanticType.NUMERIC) for c in "abcd"]
    scarce = profile(30, numeric)
    ridge = PACK.score(scarce, TaskType.REGRESSION)
    assert ridge.score > LINEAR_REGRESSION.score(scarce, TaskType.REGRESSION).score
