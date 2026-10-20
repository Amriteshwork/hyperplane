import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml.algorithms.linear_regression import PACK
from app.ml.base import Progress
from app.ml.preprocessing import PreprocessingConfig
from app.ml.training import STAGES, TrainingConfig, codegen_context, dump_model, train
from app.ml.types import TaskType

RNG = np.random.default_rng(0)
DF = pd.DataFrame(
    {
        "a": RNG.normal(size=60),
        "b": RNG.normal(size=60),
        "segment": RNG.choice(["x", "y"], size=60),
    }
)
DF["target"] = 3 * DF["a"] - 2 * DF["b"] + (DF["segment"] == "y") * 5
DF.loc[[3, 7], "target"] = None  # rows without a target are dropped
CONFIG = TrainingConfig(
    target="target",
    task_type=TaskType.REGRESSION,
    preprocessing=PreprocessingConfig(numeric=("a", "b"), categorical=("segment",)),
    test_size=0.25,
    seed=7,
    params=PACK.params_model().model_dump(mode="json"),
)


def test_train_fits_evaluates_and_reports_progress() -> None:
    progress: list[Progress] = []
    result = train(DF, CONFIG, PACK, progress.append)

    assert [p.message for p in progress] == list(STAGES)
    assert (result.report.n_train, result.report.n_test) == (43, 15)
    r2 = next(m for m in result.report.evaluation.metrics if m.id == "r2")
    assert r2.value == pytest.approx(1.0)
    coefficients = result.report.coefficients
    assert {c.feature for c in coefficients} == {
        "numeric__a",
        "numeric__b",
        "categorical__segment_x",
        "categorical__segment_y",
    }
    assert [abs(c.value) for c in coefficients] == sorted(
        (abs(c.value) for c in coefficients), reverse=True
    )
    assert len(result.predictions) == len(result.X_test) == 15


def test_none_in_object_columns_counts_as_missing() -> None:
    # Parquet returns None (not NaN) for gaps in object columns, e.g. booleans with nulls.
    df = DF.assign(flag=pd.Series([True, False, None] * 20, dtype=object))
    config = CONFIG.model_copy(
        update={"preprocessing": PreprocessingConfig(numeric=("a",), categorical=("flag",))}
    )
    result = train(df, config, PACK)
    assert {c.feature for c in result.report.coefficients} == {
        "numeric__a",
        "categorical__flag_False",
        "categorical__flag_True",
    }


def test_same_seed_same_split() -> None:
    first, second = train(DF, CONFIG, PACK), train(DF, CONFIG, PACK)
    assert first.X_test.index.equals(second.X_test.index)


def test_model_round_trips_through_joblib(tmp_path: object) -> None:
    import io

    result = train(DF, CONFIG, PACK)
    loaded = joblib.load(io.BytesIO(dump_model(result.pipeline)))  # our own artefact
    np.testing.assert_allclose(loaded.predict(result.X_test), result.predictions)


def test_codegen_context_mirrors_the_config() -> None:
    context = codegen_context(CONFIG, "data.csv")
    assert context.numeric_features == ["a", "b"]
    assert context.categorical_features == ["segment"]
    assert (context.target, context.test_size, context.seed) == ("target", 0.25, 7)
    assert context.params == {"fit_intercept": True, "positive": False}
