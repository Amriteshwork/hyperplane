"""Pack-specific checks for MLP Classifier (the shared contract lives in tests/contract)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

from app.ml.algorithms.mlp_classifier import PACK
from app.ml.algorithms.mlp_params import mlp_kwargs
from app.ml.base import TaskType
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType
from app.ml.training import TrainingConfig, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mlp_classifier_scratch", Path("app/ml/algorithms/mlp_classifier/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classes(n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=400,
        n_features=5,
        n_informative=3,
        n_classes=n_classes,
        flip_y=0.1,
        random_state=0,
    )
    return X, np.array(["ant", "bee", "cat"])[y]  # string labels, as in a CSV


@pytest.mark.parametrize("n_classes", [2, 3])  # a sigmoid output, then a softmax
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # capped on purpose
@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic"])
def test_scratch_replays_scikit_learns_training_exactly(
    scratch: ModuleType, n_classes: int, activation: str
) -> None:
    X, y = classes(n_classes)
    settings = {"hidden_layer_sizes": (16,), "activation": activation, "max_iter": 300}
    ours = scratch.MLPClassifierScratch(**settings, random_state=0).fit(X, y)
    theirs = MLPClassifier(**settings, random_state=0).fit(X, y)

    assert ours.n_iter_ == theirs.n_iter_
    np.testing.assert_array_equal(ours.classes_, theirs.classes_)
    np.testing.assert_allclose(ours.loss_curve_, theirs.loss_curve_, rtol=1e-12)
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-12)
    np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # capped on purpose
def test_output_layer_follows_the_number_of_classes() -> None:
    X, y = classes(2)
    binary = PACK.build_estimator(PACK.params_model(max_iter=50)).fit(X, y)
    assert binary.n_outputs_ == 1 and binary.out_activation_ == "logistic"
    X3, y3 = classes(3)
    multi = PACK.build_estimator(PACK.params_model(max_iter=50)).fit(X3, y3)
    assert multi.n_outputs_ == 3 and multi.out_activation_ == "softmax"


def test_defaults_are_scikit_learns_but_a_looser_stop_and_a_fixed_seed() -> None:
    ours = mlp_kwargs(PACK.params_model())
    theirs = MLPClassifier().get_params()
    assert {k for k in ours if ours[k] != theirs[k]} == {
        "hidden_layer_sizes",  # [100] for (100,)
        "max_iter",
        "tol",
        "random_state",
    }


def test_runs_report_probabilities_and_a_validation_accuracy_curve() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["label"] = np.where(df["a"] ** 2 + df["b"] ** 2 > 1.4, "out", "in")  # a circle
    config = TrainingConfig(
        target="label",
        task_type=TaskType.BINARY,
        preprocessing=PreprocessingConfig(numeric=("a", "b")),
        test_size=0.25,
        seed=0,
        params=PACK.params_model(early_stopping=True, validation_fraction=0.2).model_dump(
            mode="json"
        ),
    )
    result = train(df, config, PACK)
    assert result.probabilities is not None
    plots = {p.id: p for p in result.report.evaluation.plots}
    assert {"confusion-matrix", "roc", "loss-curve", "validation-curve"} <= set(plots)
    assert plots["validation-curve"].y_label == "Validation accuracy"
    assert result.report.coefficients == []  # a network has no coefficients to report
    assert result.report.network is not None
    assert result.report.network.layers[-1].shown == ["P(out)"]  # one logistic output
    assert len(result.report.feature_importance) == 2


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # capped on purpose
def test_multiclass_networks_have_an_output_per_class() -> None:
    X, y = classes(3)
    model = PACK.build_estimator(PACK.params_model(max_iter=50)).fit(X, y)
    network = PACK.network(model, [f"x{i}" for i in range(5)])
    assert network.output_names == ["ant", "bee", "cat"]
    assert network.output_activation == "softmax"


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
    small = PACK.score(profile(300), TaskType.BINARY)
    large = PACK.score(profile(50_000), TaskType.BINARY)
    assert large.score > small.score > 0
    assert any("Logistic Regression" in r for r in small.reasons)
    assert any("softmax" in r for r in PACK.score(profile(300), TaskType.MULTICLASS).reasons)
    assert any(r.startswith("When not to use") for r in large.reasons)

    refused = PACK.score(profile(300), TaskType.REGRESSION)
    assert refused.score == 0 and "MLP Regressor" in refused.reasons[0]
