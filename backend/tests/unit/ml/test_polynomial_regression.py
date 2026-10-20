"""Pack-specific checks for Polynomial Regression (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from app.ml.algorithms.polynomial_regression import PACK
from app.ml.algorithms.polynomial_regression.params import PolynomialRegressionParams
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "poly_scratch", Path("app/ml/algorithms/polynomial_regression/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 1 + X[:, 0] - 2 * X[:, 1] ** 2 + X[:, 0] * X[:, 2] + rng.normal(scale=0.1, size=200)
    return X, y


@pytest.mark.parametrize(("degree", "interaction_only"), [(1, False), (2, False), (3, True)])
def test_scratch_expansion_matches_scikit_learn(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], degree: int, interaction_only: bool
) -> None:
    X, _ = data
    theirs = PolynomialFeatures(degree, interaction_only=interaction_only, include_bias=False)
    np.testing.assert_allclose(
        scratch.polynomial_features(X, degree, interaction_only), theirs.fit_transform(X)
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_scratch_matches_the_platform_estimator(
    scratch: ModuleType, data: tuple[np.ndarray, np.ndarray], degree: int
) -> None:
    X, y = data
    ours = scratch.PolynomialRegressionScratch(degree).fit(X, y)
    theirs = PACK.build_estimator(PolynomialRegressionParams(degree=degree)).fit(X, y)
    regress = theirs.named_steps["regress"]

    np.testing.assert_allclose(ours.coef_, regress.coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept_ == pytest.approx(regress.intercept_, abs=1e-8)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)


def test_degree_one_is_linear_regression(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    poly = PACK.build_estimator(PolynomialRegressionParams(degree=1)).fit(X, y)
    np.testing.assert_allclose(poly.predict(X), LinearRegression().fit(X, y).predict(X))


def test_degree_sweep_shows_overfitting(scratch: ModuleType) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=40)
    y = np.sin(3 * x) + rng.normal(scale=0.2, size=40)
    sweep = {degree: (train, test) for degree, train, test in scratch.degree_sweep(x, y)}

    train_errors = [train for train, _ in sweep.values()]
    assert train_errors == sorted(train_errors, reverse=True)  # more terms never fit worse
    best = min(sweep, key=lambda d: sweep[d][1])
    assert 1 < best < 12  # neither the line nor the wiggliest curve generalises best
    assert sweep[12][1] > sweep[best][1]


def test_coefficients_are_named_by_term(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    fitted = PACK.build_estimator(PolynomialRegressionParams()).fit(X, y)
    table = PACK.coefficients(fitted, ["a", "b", "c"])

    assert list(table) == ["a", "b", "c", "a^2", "a b", "a c", "b^2", "b c", "c^2"]
    assert table["b^2"] == pytest.approx(-2, abs=0.05)
    assert table["a c"] == pytest.approx(1, abs=0.05)


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
    return DatasetProfile(n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns)


def test_recommendation_refuses_classification() -> None:
    result = PACK.score(profile(100, [column("a", SemanticType.NUMERIC)]), TaskType.BINARY)
    assert result.score == 0
    assert "Logistic Regression" in result.reasons[0]


def test_recommendation_penalises_term_explosion() -> None:
    few = [column(c, SemanticType.NUMERIC) for c in "abcd"]
    many = [column(f"x{i}", SemanticType.NUMERIC) for i in range(30)]
    small = PACK.score(profile(5000, few), TaskType.REGRESSION)
    wide = PACK.score(profile(5000, many), TaskType.REGRESSION)

    assert small.score > wide.score
    assert any("Ridge" in r for r in wide.reasons)  # 29 inputs → 464 terms, ~11 rows each
    assert any(r.startswith("When not to use") for r in small.reasons)


def test_recommendation_counts_one_hot_columns() -> None:
    columns = [column("y", SemanticType.NUMERIC), column("city", SemanticType.CATEGORICAL, 5)]
    result = PACK.score(profile(1000, columns), TaskType.REGRESSION)
    assert any("~5 input column(s) become ~20 terms" in r for r in result.reasons)
