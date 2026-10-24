"""Pack-specific checks for MLP Regressor (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.mlp_params import mlp_kwargs
from app.ml.algorithms.mlp_regressor import PACK
from app.ml.base import TaskType
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType
from app.ml.training import TrainingConfig, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mlp_regressor_scratch", Path("app/ml/algorithms/mlp_regressor/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def curved() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=500, n_features=4, noise=5, random_state=0)
    y = np.sin(y / 50)  # non-linear in the features
    return StandardScaler().fit_transform(X), (y - y.mean()) / y.std()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # capped on purpose
@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic", "identity"])
@pytest.mark.parametrize("max_iter", [20, 1000])  # capped, and stopped by its own rule
def test_scratch_replays_scikit_learns_training_exactly(
    scratch: ModuleType, activation: str, max_iter: int
) -> None:
    # Same initial draws, shuffles, mini-batches and Adam steps: non-convex, yet identical.
    X, y = curved()
    settings = {"hidden_layer_sizes": (16, 8), "activation": activation, "max_iter": max_iter}
    ours = scratch.MLPRegressorScratch(**settings, random_state=0).fit(X, y)
    theirs = MLPRegressor(**settings, random_state=0).fit(X, y)

    assert ours.n_iter_ == theirs.n_iter_
    np.testing.assert_allclose(ours.loss_curve_, theirs.loss_curve_, rtol=1e-12)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), atol=1e-12)


def test_shape_params_map_to_hidden_layer_sizes() -> None:
    params = PACK.params_model(hidden_layers=3, hidden_units=8)
    kwargs = mlp_kwargs(params)
    assert kwargs["hidden_layer_sizes"] == [8, 8, 8]
    assert kwargs["batch_size"] == "auto"
    assert "hidden_layers" not in kwargs and "hidden_units" not in kwargs
    assert mlp_kwargs(PACK.params_model(batch_size=32))["batch_size"] == 32


def test_defaults_are_scikit_learns_but_a_looser_stop_and_a_fixed_seed() -> None:
    ours = mlp_kwargs(PACK.params_model())
    theirs = MLPRegressor().get_params()
    changed = {k for k in ours if ours[k] != theirs[k]}
    assert changed == {"hidden_layer_sizes", "max_iter", "tol", "random_state"}
    assert ours["hidden_layer_sizes"] == [100]  # scikit-learn's (100,), as a list
    assert ours["max_iter"] == 1000


def test_params_reject_values_scikit_learn_would() -> None:
    with pytest.raises(ValidationError):
        PACK.params_model(hidden_units=0)
    with pytest.raises(ValidationError):
        PACK.params_model(activation="softmax")  # an output activation, not a hidden one
    with pytest.raises(ValidationError):
        PACK.params_model(validation_fraction=1.0)


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    df["y"] = 1000 * np.sin(df["a"]) + 300 * df["b"] ** 2 + rng.normal(scale=50, size=200)
    return df


def config(**params: object) -> TrainingConfig:
    return TrainingConfig(
        target="y",
        task_type=TaskType.REGRESSION,
        preprocessing=PreprocessingConfig(numeric=("a", "b")),
        test_size=0.25,
        seed=0,
        params=PACK.params_model(**params).model_dump(mode="json"),
    )


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # capped on purpose
def test_runs_report_their_loss_curve_and_validation_scores() -> None:
    plain = train(frame(), config(), PACK).report.evaluation.plots
    curve = next(p for p in plain if p.id == "loss-curve")
    assert curve.axes == "data" and curve.series[0].points[0][0] == 1  # epochs count from 1
    assert "validation-curve" not in [p.id for p in plain]

    stopped = train(frame(), config(early_stopping=True), PACK).report.evaluation.plots
    validation = next(p for p in stopped if p.id == "validation-curve")
    assert validation.y_label == "Validation R²"

    lbfgs = train(frame(), config(solver="lbfgs", max_iter=200), PACK).report.evaluation.plots
    assert {"loss-curve", "validation-curve"}.isdisjoint(p.id for p in lbfgs)  # no epochs


def test_target_is_standardised_and_predictions_come_back_in_its_units() -> None:
    result = train(frame(), config(), PACK)
    wrapper = result.pipeline.named_steps["model"]
    assert wrapper.transformer_.scale_[0] > 100  # trained on y / s_y, not on y
    assert abs(result.predictions.mean() - frame()["y"].mean()) < 300
    assert result.report.evaluation.metrics[0].value > 0.8  # R²: the curve was learned


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


def test_recommendation_prefers_large_data_and_refuses_classification() -> None:
    small = PACK.score(profile(300), TaskType.REGRESSION)
    large = PACK.score(profile(50_000), TaskType.REGRESSION)
    assert large.score > small.score > 0
    assert any("Ridge" in r for r in small.reasons)
    assert any(r.startswith("When not to use") for r in large.reasons)

    refused = PACK.score(profile(300), TaskType.BINARY)
    assert refused.score == 0 and "MLP Classifier" in refused.reasons[0]
