"""Pack-specific checks for KNN Regressor (the shared contract lives in tests/contract)."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.knn_inspect import MAX_CURVE_ROWS, k_curve
from app.ml.algorithms.knn_regressor import PACK
from app.ml.base import TaskType
from app.ml.codegen import render
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import ColumnProfile, SemanticType
from app.ml.training import TrainingConfig, codegen_context, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "knn_regressor_scratch", Path("app/ml/algorithms/knn_regressor/scratch.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def curved(rows: int = 300, noise: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.uniform(-3, 3, rows), "b": rng.uniform(-3, 3, rows)})
    df["y"] = np.sin(df["a"]) * 10 + df["b"] ** 2 + rng.normal(scale=noise, size=rows)
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


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("k", [1, 5, 40])
def test_scratch_matches_scikit_learn_exactly(
    scratch: ModuleType, weights: str, metric: str, k: int
) -> None:
    df = curved()
    X, y, query = df[["a", "b"]].to_numpy()[:200], df["y"].to_numpy()[:200], df[["a", "b"]][200:]
    settings = {"n_neighbors": k, "weights": weights, "metric": metric}
    ours = scratch.KNNRegressorScratch(**settings).fit(X, y)
    theirs = KNeighborsRegressor(**settings, algorithm="brute").fit(X, y)

    np.testing.assert_allclose(ours.predict(query.to_numpy()), theirs.predict(query.to_numpy()))
    distances, rows = ours.kneighbors(query.to_numpy())
    their_distances, their_rows = theirs.kneighbors(query.to_numpy())
    np.testing.assert_allclose(distances, their_distances, atol=1e-12)
    np.testing.assert_array_equal(rows, their_rows)


def test_scratch_lets_an_identical_row_decide_alone(scratch: ModuleType) -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([10.0, 20.0, 30.0])
    ours = scratch.KNNRegressorScratch(n_neighbors=3, weights="distance").fit(X, y)
    theirs = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(X, y)
    query = np.array([[1.0], [0.5]])
    np.testing.assert_allclose(ours.predict(query), theirs.predict(query))
    assert ours.predict(query)[0] == 20.0


def test_defaults_are_scikit_learns() -> None:
    ours = PACK.build_estimator(PACK.params_model()).get_params()
    theirs = KNeighborsRegressor().get_params()
    changed = {k for k in theirs if ours[k] != theirs[k]}
    assert changed == {"metric"}  # euclidean is minkowski with p = 2, by its own name
    df = curved()
    X, y = df[["a", "b"]], df["y"]
    assert (
        KNeighborsRegressor().fit(X, y).predict(X.head()).tolist()
        == PACK.build_estimator(PACK.params_model()).fit(X, y).predict(X.head()).tolist()
    )


@pytest.mark.parametrize("bad", [{"n_neighbors": 0}, {"weights": "gaussian"}, {"p": 3}])
def test_invalid_parameters_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PACK.params_model(**bad)


def test_training_only_stores_the_rows() -> None:
    df = curved()
    model = make_pipeline(StandardScaler(), PACK.build_estimator(PACK.params_model()))
    model.fit(df[["a", "b"]], df["y"])
    assert model[-1].n_samples_fit_ == len(df)


def test_averaging_neighbours_beats_copying_the_nearest_on_noisy_data() -> None:
    def r2(k: int) -> float:
        metrics = train(curved(noise=4.0), config(n_neighbors=k), PACK).report.evaluation.metrics
        return {m.label: m.value for m in metrics}["R²"]

    assert r2(5) > r2(1)


def test_runs_report_a_cross_validated_score_per_k() -> None:
    result = train(curved(), config(n_neighbors=7), PACK)
    plots = {p.id: p for p in result.report.evaluation.plots}
    curve = plots["tuning-curve"]
    assert curve.y_label == "Cross-validated R²"
    assert curve.x_label == "Neighbours (k)"
    assert [x for x, _ in curve.series[0].points] == [1, 3, 5, 7, 10, 20, 50]
    assert curve.marker is not None and curve.marker.x == 7  # this run's k joins the candidates
    assert result.report.coefficients == []


def test_the_script_prints_the_scores_the_platform_plots(tmp_path: Path) -> None:
    df = curved()
    df.to_csv(tmp_path / "data.csv", index=False)
    settings = config(n_neighbors=7)
    result = train(df, settings, PACK)
    code = render(PACK.templates()["train"], codegen_context(settings, "data.csv")).code
    (tmp_path / "train.py").write_text(code)
    script = subprocess.run(
        [sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    printed = {
        float(k): float(v)
        for k, v in re.findall(r"k = +(\d+): cross-validated R² (-?[\d.]+)", script.stdout)
    }
    plot = next(p for p in result.report.evaluation.plots if p.id == "tuning-curve")
    assert printed == pytest.approx(dict(plot.series[0].points), abs=5e-4)
    assert "Its neighbours:" in script.stdout
    nearest, typical = re.search(
        r"k-th neighbour is ([\d.]+) away on average; a typical stored row, ([\d.]+)",
        script.stdout,
    ).groups()
    assert result.report.neighbours is not None
    assert result.report.neighbours.nearest_distance == pytest.approx(float(nearest), abs=5e-4)
    assert result.report.neighbours.typical_distance == pytest.approx(float(typical), abs=5e-4)


def test_the_curve_stops_at_the_largest_k_a_fold_can_serve() -> None:
    df = curved(rows=30)  # 22 training rows: each fold trains on 17
    result = train(df, config(), PACK)
    plot = next(p for p in result.report.evaluation.plots if p.id == "tuning-curve")
    assert [x for x, _ in plot.series[0].points] == [1, 3, 5, 10]


def test_the_curve_is_skipped_when_it_would_take_too_long() -> None:
    rows = MAX_CURVE_ROWS + 1
    pipeline = Pipeline([("model", PACK.build_estimator(PACK.params_model()))])
    X = pd.DataFrame({"a": np.zeros(rows)})
    assert k_curve(pipeline, X, np.zeros(rows), "score") is None


def test_runs_trace_their_biggest_misses_and_best_predictions_to_stored_rows() -> None:
    report = train(curved(), config(n_neighbors=3), PACK).report.neighbours
    assert report is not None
    assert (report.k, report.stored_rows, report.total_columns) == (3, 225, 2)
    assert set(report.columns) == {"a", "b"}
    assert [e.kind for e in report.examples] == ["miss"] * 3 + ["hit"] * 3
    error = [abs(float(e.predicted) - float(e.actual)) for e in report.examples]
    assert error[0] >= error[2] > error[3]  # worst first, then the best
    first = report.examples[0]
    assert len(first.neighbours) == 3
    assert [n.distance for n in first.neighbours] == sorted(n.distance for n in first.neighbours)
    assert sum(n.share for n in first.neighbours) == pytest.approx(1, abs=1e-4)
    # The prediction really is what these rows say.
    assert float(first.predicted) == pytest.approx(
        np.mean([float(n.target) for n in first.neighbours]), rel=1e-2
    )
    assert 0 < report.nearest_distance < report.typical_distance


def test_closer_neighbours_get_a_larger_share_with_distance_weights() -> None:
    report = train(curved(), config(weights="distance"), PACK).report.neighbours
    assert report is not None
    shares = [n.share for n in report.examples[0].neighbours]
    assert shares == sorted(shares, reverse=True) and shares[0] > shares[-1]
    assert sum(shares) == pytest.approx(1, abs=1e-4)


def test_a_few_test_rows_are_each_shown_once() -> None:
    report = train(curved(rows=16), config(), PACK).report.neighbours
    assert report is not None
    assert len(report.examples) == 4  # 25% of 16, not 3 + 3
    assert [e.kind for e in report.examples] == ["miss"] * 3 + ["hit"]


def test_the_best_k_is_offered_for_the_next_run() -> None:
    tuning = train(curved(noise=4.0), config(n_neighbors=1), PACK).report.tuning
    assert tuning is not None
    assert (tuning.name, tuning.chosen) == ("n_neighbors", 1)
    assert tuning.best > 1
    PACK.params_model(n_neighbors=int(tuning.best))  # what the Lab will send


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_every_metric_trains_on_sparse_one_hot_data(metric: str) -> None:
    rng = np.random.default_rng(0)
    df = curved(rows=400)
    df["c1"], df["c2"] = [[f"{name}{i}" for i in rng.integers(0, 60, 400)] for name in ("c", "d")]
    df["flag"] = rng.integers(0, 2, 400).astype(bool)
    df.loc[::7, "a"] = np.nan
    settings = config(metric=metric).model_copy(
        update={
            "preprocessing": PreprocessingConfig(
                numeric=("a", "b"), categorical=("c1", "c2", "flag")
            )
        }
    )
    report = train(df, settings, PACK).report.neighbours
    assert report is not None and report.total_columns == 5
    assert (
        "missing"
        in str(
            [e.values for e in report.examples]
            + [n.values for e in report.examples for n in e.neighbours]
        )
        or "a" not in report.columns
    )


def test_k_above_the_training_rows_fails_with_a_readable_message() -> None:
    with pytest.raises(ValueError, match="n_neighbors <= n_samples_fit"):
        train(curved(rows=20), config(n_neighbors=500), PACK)


def profile(rows: int, features: int = 4) -> DatasetProfile:
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
        for i in range(features)
    ]
    return DatasetProfile(
        n_rows=rows, n_columns=len(columns), duplicate_rows=0, columns=columns, correlations=None
    )


def test_recommendation_prefers_few_features_and_modest_rows_and_refuses_classes() -> None:
    good = PACK.score(profile(5_000), TaskType.REGRESSION)
    wide = PACK.score(profile(5_000, features=60), TaskType.REGRESSION)
    huge = PACK.score(profile(90_000), TaskType.REGRESSION)
    sparse = PACK.score(profile(20), TaskType.REGRESSION)
    assert good.score > wide.score > 0
    assert good.score > huge.score > 0
    assert good.score > sparse.score > 0
    assert any("curse of dimensionality" in r for r in wide.reasons)
    assert any("slow" in r for r in huge.reasons)
    assert any(r.startswith("When not to use") for r in good.reasons)

    refused = PACK.score(profile(300), TaskType.BINARY)
    assert refused.score == 0 and "KNN Classifier" in refused.reasons[0]
