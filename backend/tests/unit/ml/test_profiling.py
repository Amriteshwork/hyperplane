from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from app.ml.profiling import SemanticType, infer_task_type, profile_dataframe
from app.ml.types import TaskType
from tests.conftest import DATASETS_DIR

SAMPLES = sorted(p.parent for p in DATASETS_DIR.glob("*/data.csv"))


@pytest.mark.parametrize("folder", SAMPLES, ids=[p.name for p in SAMPLES])
def test_profiles_every_sample(folder: Path) -> None:
    meta = yaml.safe_load((folder / "meta.yaml").read_text())
    df = pd.read_csv(folder / "data.csv")

    profile = profile_dataframe(df)

    assert (profile.n_rows, profile.n_columns) == (meta["rows"], meta["columns"])
    assert [c.name for c in profile.columns] == list(df.columns)
    # The documented target is suggested first, with the documented task type.
    first = profile.target_candidates[0]
    assert (first.column, first.task_type) == (meta["target"], TaskType(meta["task"]))
    assert infer_task_type(df[meta["target"]]) == TaskType(meta["task"])
    # JSON round-trip: no NaN/inf leaks into the payload.
    assert profile.model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.parametrize("folder", SAMPLES, ids=[p.name for p in SAMPLES])
def test_numeric_columns_have_summaries_and_histograms(folder: Path) -> None:
    profile = profile_dataframe(pd.read_csv(folder / "data.csv"))
    for column in profile.columns:
        if column.semantic_type is SemanticType.NUMERIC:
            assert column.numeric is not None and column.numeric.histogram is not None
            hist = column.numeric.histogram
            assert len(hist.edges) == len(hist.counts) + 1
            assert sum(hist.counts) == profile.n_rows - column.missing


def test_penguins_missing_values_and_categoricals() -> None:
    profile = profile_dataframe(pd.read_csv(DATASETS_DIR / "penguins" / "data.csv"))
    by_name = {c.name: c for c in profile.columns}
    assert by_name["sex"].missing == 11
    assert by_name["sex"].missing_pct == pytest.approx(3.2, abs=0.01)
    assert by_name["island"].semantic_type is SemanticType.CATEGORICAL
    assert by_name["island"].categorical is not None
    assert {v.value for v in by_name["island"].categorical.top} == {"Biscoe", "Dream", "Torgersen"}


def test_semantic_types() -> None:
    n = 60
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "row": range(n),
            "flag": [0, 1] * (n // 2),
            "answer": ["yes", "no"] * (n // 2),
            "signup": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "comment": [
                "this is a fairly long free text comment number " + str(i) for i in range(n)
            ],
            "colour": ["red", "green", "blue"] * (n // 3),
            "amount": np.linspace(0, 1, n),
        }
    )
    types = {c.name: c.semantic_type for c in profile_dataframe(df).columns}
    assert types == {
        "customer_id": SemanticType.ID_LIKE,
        "row": SemanticType.ID_LIKE,
        "flag": SemanticType.BOOLEAN,
        "answer": SemanticType.BOOLEAN,
        "signup": SemanticType.DATETIME,
        "comment": SemanticType.TEXT,
        "colour": SemanticType.CATEGORICAL,
        "amount": SemanticType.NUMERIC,
    }


def test_correlations_are_json_safe() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "c": [5.0] * 4})
    corr = profile_dataframe(df).correlations
    assert corr is not None and corr.columns == ["a", "b", "c"]
    assert corr.values[0][1] == pytest.approx(1.0)
    assert corr.values[0][2] is None  # constant column → undefined correlation


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.25, 3.0, 4.75], TaskType.REGRESSION),
        (list(range(100)) * 2, TaskType.REGRESSION),
        ([0, 1, 0, 1], TaskType.BINARY),
        (["a", "b", "c", "a"], TaskType.MULTICLASS),
        ([1, 2, 3, 1, 2, 3], TaskType.MULTICLASS),
        ([7, 7, 7], None),
    ],
)
def test_infer_task_type(values: list[object], expected: TaskType | None) -> None:
    assert infer_task_type(pd.Series(values, name="target")) == expected


def test_empty_frame() -> None:
    profile = profile_dataframe(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    assert profile.n_rows == 0 and profile.target_candidates == []
