"""Pack-specific checks for Gaussian Naive Bayes (the shared contract lives in tests/contract)."""

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
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from app.ml.algorithms.gaussian_nb import PACK
from app.ml.algorithms.logistic_regression import PACK as LOGISTIC
from app.ml.base import TaskType
from app.ml.codegen import render
from app.ml.preprocessing import PreprocessingConfig
from app.ml.profiling import DatasetProfile
from app.ml.profiling.profile import (
    ColumnProfile,
    CorrelationMatrix,
    NumericSummary,
    SemanticType,
)
from app.ml.training import TrainingConfig, codegen_context, train


@pytest.fixture(scope="module")
def scratch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gaussian_nb_scratch", Path("app/ml/algorithms/gaussian_nb/scratch.py")
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
        flip_y=0.1,
        random_state=0,
    )
    return X, np.array(["ant", "bee", "cat"])[y]  # string labels, as in a CSV


def blobs(rows: int = 400) -> pd.DataFrame:
    """Two noisy classes: `a` separates them, `twin` nearly copies `a`, and `noise` does nothing."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, rows)
    a = rng.normal(100.0 * y, 80.0)
    return pd.DataFrame(
        {
            "a": a,
            "twin": a + rng.normal(0, 5.0, rows),
            "noise": rng.normal(size=rows),
            "label": np.array(["no", "yes"])[y],
        }
    )


def config(
    numeric: tuple[str, ...] = ("a", "twin", "noise"),
    categorical: tuple[str, ...] = (),
    **params: object,
) -> TrainingConfig:
    return TrainingConfig(
        target="label",
        task_type=TaskType.BINARY,
        preprocessing=PreprocessingConfig(numeric=numeric, categorical=categorical),
        test_size=0.25,
        seed=0,
        params=PACK.params_model(**params).model_dump(mode="json"),
    )


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("smoothing", [1e-9, 0.01, 1.0])
def test_scratch_matches_scikit_learn(
    scratch: ModuleType, n_classes: int, smoothing: float
) -> None:
    X, y = classes(n_classes)
    ours = scratch.GaussianNBScratch(var_smoothing=smoothing).fit(X, y)
    theirs = GaussianNB(var_smoothing=smoothing).fit(X, y)
    np.testing.assert_allclose(ours.class_prior_, theirs.class_prior_, atol=1e-12)
    np.testing.assert_allclose(ours.theta_, theirs.theta_, atol=1e-10)
    np.testing.assert_allclose(ours.var_, theirs.var_, atol=1e-10)
    np.testing.assert_allclose(
        ours.joint_log_proba(X), theirs.predict_joint_log_proba(X), atol=1e-8
    )
    np.testing.assert_allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-10)
    np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))


def test_scratch_defaults_match_the_params(scratch: ModuleType) -> None:
    assert scratch.GaussianNBScratch().var_smoothing == PACK.params_model().var_smoothing


def test_scratch_survives_rows_that_would_underflow(scratch: ModuleType) -> None:
    X, y = classes(2)
    probabilities = scratch.GaussianNBScratch(var_smoothing=1e-9).fit(X, y).predict_proba(X * 1e4)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_params_reject_values_the_estimator_cannot_use() -> None:
    for value in (0, -1, 11):
        with pytest.raises(ValidationError):
            PACK.params_model(var_smoothing=value)
    with pytest.raises(ValidationError):
        PACK.params_model(priors=[0.5, 0.5])


def test_estimator_gets_the_settings() -> None:
    model = PACK.build_estimator(PACK.params_model(var_smoothing=0.3))
    assert isinstance(model, GaussianNB)
    assert model.get_params()["var_smoothing"] == 0.3


def test_many_one_hot_columns_train_though_the_estimator_rejects_sparse_input() -> None:
    rng = np.random.default_rng(0)
    df = blobs()
    # Well below the density at which a ColumnTransformer switches to sparse output.
    df["c1"], df["c2"] = [[f"{n}{i}" for i in rng.integers(0, 60, 400)] for n in ("c", "d")]
    df.loc[::7, "a"] = np.nan
    result = train(df, config(("a",), ("c1", "c2")), PACK)
    assert result.report.gaussians is not None
    assert result.report.gaussians.total_features > 100


def test_categorical_columns_alone_train() -> None:
    rng = np.random.default_rng(0)
    df = blobs()
    df["c1"] = [f"c{i}" for i in rng.integers(0, 60, 400)]
    report = train(df, config((), ("c1",)), PACK).report.gaussians
    assert report is not None
    assert report.correlated is None  # no numeric columns to compare


def test_the_bell_curves_are_in_the_datas_own_units_and_best_separators_come_first() -> None:
    df = blobs()
    report = train(df, config(var_smoothing=1e-9), PACK).report.gaussians
    assert report is not None
    assert report.classes == ["no", "yes"]
    assert report.total_features == 3
    assert sum(report.priors) == pytest.approx(1, abs=1e-4)
    assert report.features[-1].feature == "numeric__noise"
    by_name = {f.feature: f for f in report.features}
    a = by_name["numeric__a"]
    assert a.means[1] - a.means[0] == pytest.approx(100, abs=25)  # not in standard deviations
    assert a.stds[0] == pytest.approx(80, abs=15)


def test_a_predictions_sum_is_the_models_own_score() -> None:
    df = blobs()
    settings = config()
    result = train(df, settings, PACK)
    report = result.report.gaussians
    assert report is not None
    assert [e.kind for e in report.examples] == ["miss"] * 3 + ["hit"] * 3
    assert [e.confidence for e in report.examples[3:]] == sorted(
        (e.confidence for e in report.examples[3:]), reverse=True
    )
    for example in report.examples:
        assert example.rival != example.predicted
        assert example.others is None  # three columns all fit
        assert example.evidence[0].feature in {"numeric__a", "numeric__twin"}
        total = example.prior.predicted + sum(e.predicted for e in example.evidence)
        rival = example.prior.rival + sum(e.rival for e in example.evidence)
        assert total >= rival  # the prediction is the larger sum
        assert total - rival == pytest.approx(
            np.log(example.confidence / (1 - example.confidence)), abs=1e-2
        )
    for miss in (e for e in report.examples if e.predicted != e.actual):
        assert miss.rival == miss.actual


def test_columns_past_the_limit_are_summed_so_the_totals_stay_real() -> None:
    rng = np.random.default_rng(0)
    df = blobs()
    names = [f"x{i}" for i in range(7)]
    for name in names:
        df[name] = rng.normal(size=400)
    report = train(df, config(("a", *names)), PACK).report.gaussians
    assert report is not None
    example = report.examples[0]
    assert len(example.evidence) == 5
    assert example.others is not None
    assert example.others.feature == "3 other columns"
    total = example.prior.predicted + sum(e.predicted for e in example.evidence)
    rival = example.prior.rival + sum(e.rival for e in example.evidence)
    total, rival = total + example.others.predicted, rival + example.others.rival
    assert total - rival == pytest.approx(
        np.log(example.confidence / (1 - example.confidence)), abs=1e-2
    )


def test_duplicated_evidence_is_found_and_shows_up_as_overconfidence() -> None:
    df = blobs(2000)
    report = train(df, config(), PACK).report.gaussians
    assert report is not None
    assert report.correlated is not None
    assert {report.correlated.a, report.correlated.b} == {"numeric__a", "numeric__twin"}
    assert report.correlated.correlation > 0.9
    assert report.mean_confidence > report.accuracy
    without = train(df, config(("a", "noise")), PACK).report.gaussians
    assert without is not None
    assert without.mean_confidence < report.mean_confidence


@pytest.mark.filterwarnings("ignore:The least populated class")
def test_a_few_test_rows_are_each_explained_once() -> None:
    report = train(blobs(rows=16), config(), PACK).report.gaussians
    assert report is not None
    assert [e.kind for e in report.examples] == ["miss"] * 3 + ["hit"]


def test_a_class_never_seen_in_training_is_compared_with_the_runner_up() -> None:
    df = blobs()
    result = train(df, config(), PACK)
    features = ["a", "twin", "noise"]
    train_rows, test_rows = df.iloc[:300], df.iloc[300:]
    y_test = test_rows["label"].copy()
    y_test.iloc[:2] = "new"
    report = PACK.class_gaussians(
        result.pipeline, train_rows[features], train_rows["label"], test_rows[features], y_test
    )
    assert "not seen in training" in [e.actual for e in report.examples if e.kind == "miss"]


def test_smoothing_is_tuned_on_a_log_axis_and_offered_for_the_next_run() -> None:
    result = train(blobs(), config(var_smoothing=0.05), PACK)
    plot = next(p for p in result.report.evaluation.plots if p.id == "tuning-curve")
    assert plot.x_scale == "log"
    assert plot.marker is not None and plot.marker.x == 0.05
    xs = [x for x, _ in plot.series[0].points]
    assert xs == sorted(xs) and xs[0] == 1e-9 and 0.05 in xs
    tuning = result.report.tuning
    assert tuning is not None
    assert (tuning.name, tuning.chosen) == ("var_smoothing", 0.05)
    assert tuning.score == max(y for _, y in plot.series[0].points)
    PACK.params_model(var_smoothing=tuning.best)  # what the Lab will send


def test_tiny_smoothing_lets_one_constant_column_decide(scratch: ModuleType) -> None:
    # Every `home` row has flag 1, so its variance is only the smoothing; a flag of 0 rules it out.
    X = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    y = np.array(["home"] * 3 + ["away"] * 3)
    row = np.array([[0.5, 0.0]])
    spike = scratch.GaussianNBScratch(var_smoothing=1e-9).fit(X, y)
    assert spike.joint_log_proba(row)[0, list(spike.classes_).index("home")] < -1e6
    flatter = scratch.GaussianNBScratch(var_smoothing=0.5).fit(X, y)
    assert flatter.joint_log_proba(row)[0, list(flatter.classes_).index("home")] > -10


def test_other_packs_report_no_gaussians() -> None:
    report = train(blobs(), config().model_copy(update={"params": {}}), LOGISTIC).report
    assert report.gaussians is None


def test_the_script_prints_what_the_platform_reports(tmp_path: Path) -> None:
    df = blobs()
    df.to_csv(tmp_path / "data.csv", index=False)
    settings = config(var_smoothing=0.05)
    result = train(df, settings, PACK)
    code = render(PACK.templates()["train"], codegen_context(settings, "data.csv")).code
    (tmp_path / "train.py").write_text(code)
    script = subprocess.run(
        [sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    printed = {
        float(k): float(v)
        for k, v in re.findall(
            r"var_smoothing = ([\d.e+-]+): cross-validated accuracy ([\d.]+)", script.stdout
        )
    }
    plot = next(p for p in result.report.evaluation.plots if p.id == "tuning-curve")
    assert printed == pytest.approx(dict(plot.series[0].points), abs=5e-4)

    report = result.report.gaussians
    assert report is not None
    confidence, accuracy = re.search(
        r"Average confidence ([\d.]+) vs accuracy ([\d.]+)", script.stdout
    ).groups()
    assert report.mean_confidence == pytest.approx(float(confidence), abs=5e-4)
    assert report.accuracy == pytest.approx(float(accuracy), abs=5e-4)
    a, b, r = re.search(
        r"Most correlated within a class: (\S+) and (\S+), r = (-?[\d.]+)", script.stdout
    ).groups()
    assert report.correlated is not None
    assert (report.correlated.a, report.correlated.b) == (a, b)
    assert report.correlated.correlation == pytest.approx(float(r), abs=5e-4)


def profile(
    rows: int = 5_000,
    *,
    correlated: bool = False,
    skew: float = 0.0,
    categorical: bool = False,
) -> DatasetProfile:
    numeric = NumericSummary(
        min=0,
        max=1,
        mean=0.5,
        std=0.1,
        q25=0.4,
        median=0.5,
        q75=0.6,
        skew=skew,
        zeros=0,
        histogram=None,
    )
    columns = [
        ColumnProfile(
            name=f"x{i}",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            missing=0,
            missing_pct=0.0,
            unique=10,
            unique_pct=1.0,
            numeric=numeric,
        )
        for i in range(2)
    ]
    if categorical:
        columns.append(
            ColumnProfile(
                name="c",
                dtype="object",
                semantic_type=SemanticType.CATEGORICAL,
                missing=0,
                missing_pct=0.0,
                unique=3,
                unique_pct=0.1,
            )
        )
    matrix = CorrelationMatrix(columns=["x0", "x1"], values=[[1.0, 0.95], [0.95, 1.0]])
    return DatasetProfile(
        n_rows=rows,
        n_columns=len(columns),
        duplicate_rows=0,
        columns=columns,
        correlations=matrix if correlated else None,
    )


def test_recommendation_reflects_the_models_assumptions() -> None:
    plain = PACK.score(profile(), TaskType.BINARY)
    assert any(r.startswith("When not to use") for r in plain.reasons)
    assert PACK.score(profile(rows=300), TaskType.BINARY).score > plain.score
    for worse, phrase in (
        (profile(correlated=True), "overconfident"),
        (profile(skew=5.0), "skewed"),
        (profile(categorical=True), "one-hot"),
    ):
        scored = PACK.score(worse, TaskType.MULTICLASS)
        assert scored.score < plain.score
        assert any(phrase in r for r in scored.reasons)


def test_recommendation_refuses_regression() -> None:
    scored = PACK.score(profile(), TaskType.REGRESSION)
    assert scored.score == 0
    assert "Linear Regression" in scored.reasons[0]


def test_pipeline_keeps_the_standard_step_names() -> None:
    pipeline = train(blobs(), config(), PACK).pipeline
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "model"]
