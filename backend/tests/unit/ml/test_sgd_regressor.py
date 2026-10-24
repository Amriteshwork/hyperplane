"""Pack-specific checks for SGD Regressor (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.sgd_regressor import PACK
from app.ml.base import TaskType
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sgd_regressor_scratch", Path("app/ml/algorithms/sgd_regressor/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standardised() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=0)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("alpha", [0.0001, 0.01])
def test_scratch_reaches_ridges_minimum(scratch: ModuleType, alpha: float) -> None:
    # Squared error + L2 per row is Ridge's objective with α·n. SGD stops near, not at, the
    # minimum: documented tolerance 0.1% of the largest coefficient.
    X, y = standardised()
    ours = scratch.SGDRegressorScratch(alpha=alpha).fit(X, y)
    exact = Ridge(alpha=alpha * len(y)).fit(X, y)

    assert ours.n_iter_ < ours.max_iter  # stopped by tol
    scale = np.abs(exact.coef_).max()
    np.testing.assert_allclose(ours.coef_, exact.coef_, atol=1e-3 * scale)
    assert ours.intercept_ == pytest.approx(exact.intercept_, abs=1e-3 * scale)
    assert ours.loss_curve_[0] > ours.loss_curve_[-1]


def test_scratch_huber_lands_near_scikit_learn(scratch: ModuleType) -> None:
    X, y = standardised()
    ours = scratch.SGDRegressorScratch(loss="huber", epsilon=5.0).fit(X, y)
    theirs = SGDRegressor(loss="huber", epsilon=5.0, random_state=0).fit(X, y)
    # Different shuffles and a noisy optimiser: agree within 2% of the largest coefficient.
    scale = np.abs(theirs.coef_).max()
    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=0.02 * scale)


def test_too_large_a_learning_rate_explodes(scratch: ModuleType) -> None:
    X, y = standardised()
    mse = scratch.learning_rate_sweep(X, y, np.array([1e-4, 1e-2, 5.0]))
    assert mse[1] < mse[0]  # bigger steps help...
    assert not np.isfinite(mse[2]) or mse[2] > mse[0]  # ...until they overshoot


def test_defaults_are_scikit_learns_but_a_smaller_step_and_a_fixed_seed() -> None:
    ours = PACK.params_model().model_dump()
    theirs = SGDRegressor().get_params()
    assert {k for k in ours if ours[k] != theirs[k]} == {"eta0", "tol", "random_state"}
    assert ours["eta0"] < theirs["eta0"]  # robust to rows with extreme features (ADR 0018)
    assert ours["tol"] < theirs["tol"]  # so the smaller steps still reach the minimum


def test_the_target_is_standardised_and_coefficients_come_back_in_its_units() -> None:
    X, y = standardised()
    model = PACK.build_estimator(PACK.params_model()).fit(X, 1000 * y + 5000)
    exact = Ridge(alpha=1e-4 * len(y)).fit(X, 1000 * y + 5000)
    table = PACK.coefficients(model, ["a", "b", "c", "d"])
    # The units are the point here; on 200 rows the small default step stops ~1.4% short.
    scale = np.abs(exact.coef_).max()
    np.testing.assert_allclose(list(table.values()), exact.coef_, atol=0.03 * scale)


def test_params_reject_values_scikit_learn_would() -> None:
    with pytest.raises(ValidationError):
        PACK.params_model(alpha=0)  # the optimal schedule divides by α
    with pytest.raises(ValidationError):
        PACK.params_model(loss="hinge")  # a classification loss
    with pytest.raises(ValidationError):
        PACK.params_model(eta0=0)


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


def test_recommendation_prefers_large_data_and_refuses_classes() -> None:
    small = PACK.score(profile(500), TaskType.REGRESSION)
    large = PACK.score(profile(500_000), TaskType.REGRESSION)
    assert large.score > small.score > 0
    assert any("Ridge" in r for r in small.reasons)
    assert any(r.startswith("When not to use") for r in large.reasons)

    refused = PACK.score(profile(500), TaskType.BINARY)
    assert refused.score == 0 and "SGD Classifier" in refused.reasons[0]
