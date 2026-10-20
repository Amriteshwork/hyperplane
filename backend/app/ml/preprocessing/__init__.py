"""PreprocessingConfig → ColumnTransformer."""

from app.ml.preprocessing.builder import (
    FEATURE_ROLES,
    FeatureRole,
    PreprocessingConfig,
    UnsupportedFeatureError,
    assign_roles,
    build_preprocessor,
)

__all__ = [
    "FEATURE_ROLES",
    "FeatureRole",
    "PreprocessingConfig",
    "UnsupportedFeatureError",
    "assign_roles",
    "build_preprocessor",
]
