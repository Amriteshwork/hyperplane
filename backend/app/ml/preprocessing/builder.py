"""Feature roles from the profile, and the ColumnTransformer the training templates build.

`build_preprocessor` must match the templates' `preprocess` block exactly; the
golden-equivalence test compares the two.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.profiling.profile import DatasetProfile, SemanticType


class FeatureRole(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


#: How a column of each semantic type enters the model; None means it can't be a feature yet.
FEATURE_ROLES: dict[SemanticType, FeatureRole | None] = {
    SemanticType.NUMERIC: FeatureRole.NUMERIC,
    SemanticType.CATEGORICAL: FeatureRole.CATEGORICAL,
    SemanticType.BOOLEAN: FeatureRole.CATEGORICAL,
    SemanticType.DATETIME: None,
    SemanticType.TEXT: None,
    SemanticType.ID_LIKE: None,
}


class UnsupportedFeatureError(ValueError):
    """A chosen feature's semantic type has no preprocessing yet."""


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    @property
    def features(self) -> list[str]:
        """Column order of `X`, as in the templates: numeric first, then categorical."""
        return [*self.numeric, *self.categorical]


def assign_roles(profile: DatasetProfile, features: Sequence[str]) -> PreprocessingConfig:
    """Split `features` (which must exist in `profile`) into numeric and categorical."""
    columns = {c.name: c for c in profile.columns}
    roles: dict[FeatureRole, list[str]] = {role: [] for role in FeatureRole}
    for name in features:
        semantic_type = columns[name].semantic_type
        role = FEATURE_ROLES[semantic_type]
        if role is None:
            raise UnsupportedFeatureError(
                f"{name!r} is a {semantic_type} column, which can't be used as a feature yet."
            )
        roles[role].append(name)
    return PreprocessingConfig(
        numeric=tuple(roles[FeatureRole.NUMERIC]),
        categorical=tuple(roles[FeatureRole.CATEGORICAL]),
    )


def build_preprocessor(config: PreprocessingConfig) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric, list(config.numeric))
    ]
    if config.categorical:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical, list(config.categorical)))
    return ColumnTransformer(transformers)
