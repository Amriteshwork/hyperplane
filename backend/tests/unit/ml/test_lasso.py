"""Pack-specific checks for Lasso (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression

from app.ml.algorithms.lasso import PACK
from app.ml.algorithms.linear_regression import PACK as LINEAR_REGRESSION
from app.ml.algorithms.ridge import PACK as RIDGE
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, CorrelationMatrix, SemanticType

#: Tight enough that both solvers reach the exact minimum, so their answers can be compared.
EXACT = {"tol": 1e-12, "max_iter": 100_000}


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lasso_scratch", Path("app/ml/algorithms/lasso/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=8, n_informative=3, noise=10.0, random_state=0)
    return X, y


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], alpha: float, fit_intercept: bool
) -> None:
    X, y = data
    ours = scratch.LassoScratch(alpha, fit_intercept=fit_intercept, **EXACT).fit(X, y)
    theirs = Lasso(alpha=alpha, fit_intercept=fit_intercept, **EXACT).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)
    np.testing.assert_array_equal(ours.coef_ == 0, theirs.coef_ == 0)  # the same features drop
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)
    assert 1 <= ours.n_iter_ < EXACT["max_iter"]


def test_soft_threshold_snaps_small_values_to_zero(scratch: ModuleType) -> None:
    assert scratch.soft_threshold(3.0, 1.0) == 2.0
    assert scratch.soft_threshold(-3.0, 1.0) == -2.0
    assert scratch.soft_threshold(0.5, 1.0) == 0.0
    assert scratch.soft_threshold(-1.0, 1.0) == 0.0


def test_alpha_max_is_where_every_coefficient_becomes_zero(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    top = scratch.alpha_max(X, y)
    assert np.all(scratch.LassoScratch(top, **EXACT).fit(X, y).coef_ == 0)
    assert np.count_nonzero(scratch.LassoScratch(0.99 * top, **EXACT).fit(X, y).coef_) == 1


def test_path_drops_features_and_approaches_ols(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    top = scratch.alpha_max(X, y)
    path = scratch.lasso_path(X, y, top * np.logspace(0, -6, 13))
    counts = np.count_nonzero(path, axis=1)

    assert counts[0] == 0 and counts[-1] == X.shape[1]
    assert np.any((counts > 0) & (counts < X.shape[1]))  # in between, some are exactly 0
    np.testing.assert_allclose(path[-1], LinearRegression().fit(X, y).coef_, rtol=1e-3)


def test_scratch_skips_constant_columns(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=50), np.ones(50)])
    model = scratch.LassoScratch(0.1).fit(X, 2 * X[:, 0])
    assert model.coef_[1] == 0
    assert np.isfinite(model.coef_).all()


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


def numeric(n: int) -> list[ColumnProfile]:
    return [column(f"x{i}", SemanticType.NUMERIC) for i in range(n)]


def test_recommendation_refuses_classification() -> None:
    result = PACK.score(profile(100, numeric(2)), TaskType.BINARY)
    assert result.score == 0
    assert "Logistic Regression" in result.reasons[0]


def test_recommendation_favours_many_features_and_penalises_correlation() -> None:
    correlated = CorrelationMatrix(columns=["x0", "x1"], values=[[1.0, 0.95], [0.95, 1.0]])
    few = PACK.score(profile(10_000, numeric(4)), TaskType.REGRESSION)
    many = PACK.score(profile(10_000, numeric(30)), TaskType.REGRESSION)
    collinear = PACK.score(profile(10_000, numeric(4), correlated), TaskType.REGRESSION)

    assert many.score > few.score
    assert collinear.score < few.score
    assert any("Elastic Net" in r for r in collinear.reasons)
    assert any(r.startswith("When not to use") and "Ridge" in r for r in few.reasons)


def test_recommendation_warns_when_coefficients_outnumber_rows() -> None:
    result = PACK.score(profile(20, numeric(50)), TaskType.REGRESSION)
    assert any("keeps at most 20 features" in r for r in result.reasons)


def test_lasso_outranks_ols_and_ridge_on_wide_data() -> None:
    wide = profile(200, numeric(40))
    lasso = PACK.score(wide, TaskType.REGRESSION).score
    assert lasso > LINEAR_REGRESSION.score(wide, TaskType.REGRESSION).score
    assert lasso > RIDGE.score(wide, TaskType.REGRESSION).score


def test_default_alpha_keeps_some_features_on_unit_scale_targets() -> None:
    """scikit-learn's α = 1 would zero every coefficient here; our default must not."""
    X, y = make_regression(n_samples=300, n_features=8, n_informative=3, random_state=0)
    y = (y - y.mean()) / y.std()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    coef = PACK.build_estimator(PACK.params_model()).fit(X, y).coef_
    assert 0 < np.count_nonzero(coef) < X.shape[1]
