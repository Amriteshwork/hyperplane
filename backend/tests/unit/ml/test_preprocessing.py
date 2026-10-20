import numpy as np
import pandas as pd
import pytest

from app.ml.preprocessing import (
    FEATURE_ROLES,
    PreprocessingConfig,
    UnsupportedFeatureError,
    assign_roles,
    build_preprocessor,
)
from app.ml.profiling import SemanticType, profile_dataframe

DF = pd.DataFrame(
    {
        "x": [1.0, 2.0, None, 4.0] * 5,
        "flag": [True, False, np.nan, True] * 5,
        "colour": ["red", "blue", None, "red"] * 5,
        "when": pd.date_range("2026-01-01", periods=20).astype(str),
    }
)


def test_every_semantic_type_has_a_role_entry() -> None:
    assert set(FEATURE_ROLES) == set(SemanticType)


def test_roles_follow_semantic_types_and_keep_order() -> None:
    config = assign_roles(profile_dataframe(DF), ["colour", "x", "flag"])
    assert config == PreprocessingConfig(numeric=("x",), categorical=("colour", "flag"))
    assert config.features == ["x", "colour", "flag"]


def test_unsupported_types_are_rejected() -> None:
    with pytest.raises(UnsupportedFeatureError, match="'when' is a datetime column"):
        assign_roles(profile_dataframe(DF), ["x", "when"])


def test_preprocessor_imputes_scales_and_one_hot_encodes() -> None:
    config = PreprocessingConfig(numeric=("x",), categorical=("colour", "flag"))
    out = build_preprocessor(config).fit_transform(DF[config.features])

    assert not np.isnan(np.asarray(out, dtype=float)).any()
    assert out.shape == (20, 1 + 2 + 2)  # x, colour_{blue,red}, flag_{False,True}


def test_numeric_only_has_no_categorical_branch() -> None:
    preprocessor = build_preprocessor(PreprocessingConfig(numeric=("x",)))
    assert [name for name, *_ in preprocessor.transformers] == ["numeric"]
