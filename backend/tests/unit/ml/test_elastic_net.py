"""Pack-specific checks for Elastic Net (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso

from app.ml.algorithms.elastic_net import PACK
from app.ml.algorithms.lasso import PACK as LASSO
from app.ml.algorithms.linear_regression import PACK as LINEAR_REGRESSION
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, CorrelationMatrix, SemanticType

#: Tight enough that both solvers reach the exact minimum, so their answers can be compared.
EXACT = {"tol": 1e-12, "max_iter": 100_000}


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "elastic_net_scratch", Path("app/ml/algorithms/elastic_net/scratch.py")
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
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType,
    data: tuple[np.ndarray, np.ndarray],
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
) -> None:
    X, y = data
    ours = scratch.ElasticNetScratch(alpha, l1_ratio, fit_intercept, **EXACT).fit(X, y)
    theirs = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **EXACT).fit(
        X, y
    )

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)
    np.testing.assert_array_equal(ours.coef_ == 0, theirs.coef_ == 0)  # the same features drop
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)


def test_l1_ratio_one_is_the_lasso(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    ours = scratch.ElasticNetScratch(1.0, l1_ratio=1.0, **EXACT).fit(X, y)
    np.testing.assert_allclose(ours.coef_, Lasso(alpha=1.0, **EXACT).fit(X, y).coef_, atol=1e-8)


def test_correlated_features_share_their_weight(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    X = np.column_stack([x, x])  # identical columns: the Lasso's split is arbitrary
    model = scratch.ElasticNetScratch(0.1, l1_ratio=0.5, **EXACT).fit(X, 2 * x)
    assert model.coef_[0] == pytest.approx(model.coef_[1])
    assert model.coef_[0] > 0


@pytest.mark.parametrize("l1_ratio", [0.2, 0.5, 1.0])
def test_alpha_max_is_where_every_coefficient_becomes_zero(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], l1_ratio: float
) -> None:
    X, y = data
    top = scratch.alpha_max(X, y, l1_ratio)
    assert np.all(scratch.ElasticNetScratch(top, l1_ratio, **EXACT).fit(X, y).coef_ == 0)
    assert np.count_nonzero(
        scratch.ElasticNetScratch(0.99 * top, l1_ratio, **EXACT).fit(X, y).coef_
    )


def test_grid_search_scores_every_pair(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 15))
    y = X[:, :3] @ np.array([3.0, -2.0, 1.0]) + rng.normal(scale=2.0, size=60)
    alphas, ratios = np.array([1e-3, 0.1, 1.0, 100.0]), np.array([0.2, 1.0])
    (alpha, ratio), scores = scratch.grid_search(X, y, alphas, ratios)

    assert scores.shape == (4, 2)
    assert scores[list(alphas).index(alpha), list(ratios).index(ratio)] == scores.min()
    assert 1e-3 < alpha < 100.0  # neither no penalty nor a crushing one


def test_l1_ratio_is_bounded() -> None:
    for bad in (0.0, 1.5):
        with pytest.raises(ValidationError):
            PACK.params_model(l1_ratio=bad)


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


def test_recommendation_favours_correlated_and_many_features() -> None:
    correlated = CorrelationMatrix(columns=["x0", "x1"], values=[[1.0, 0.95], [0.95, 1.0]])
    plain = PACK.score(profile(10_000, numeric(4)), TaskType.REGRESSION)
    collinear = PACK.score(profile(10_000, numeric(4), correlated), TaskType.REGRESSION)
    many = PACK.score(profile(10_000, numeric(30)), TaskType.REGRESSION)

    assert collinear.score > plain.score and many.score > plain.score
    assert any("strongly correlated" in r for r in collinear.reasons)
    assert any(r.startswith("When not to use") and "one setting" in r for r in plain.reasons)


def test_recommendation_notes_it_can_keep_more_features_than_rows() -> None:
    result = PACK.score(profile(20, numeric(50)), TaskType.REGRESSION)
    assert any("unlike Lasso" in r for r in result.reasons)


def test_elastic_net_outranks_lasso_with_correlated_features_but_not_ols_on_easy_data() -> None:
    correlated = CorrelationMatrix(columns=["x0", "x1"], values=[[1.0, 0.95], [0.95, 1.0]])
    grouped = profile(500, numeric(12), correlated)
    easy = profile(10_000, numeric(4))
    assert (
        PACK.score(grouped, TaskType.REGRESSION).score
        > LASSO.score(grouped, TaskType.REGRESSION).score
    )
    assert (
        PACK.score(easy, TaskType.REGRESSION).score
        < LINEAR_REGRESSION.score(easy, TaskType.REGRESSION).score
    )
