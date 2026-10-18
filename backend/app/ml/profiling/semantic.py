"""Semantic type detection: an ordered registry of rules, first match wins.

Order matters: a 0/1 column is boolean before numeric; an all-unique integer `id` column is
id-like before numeric; long free-form strings are text before categorical.
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

from app.ml.profiling.profile import SemanticType

_BOOL_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f"})
_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key)$|^(id|uuid|guid)_", re.IGNORECASE)
_DATE_SHAPE = re.compile(r"\d{1,4}[-/.:]\d{1,2}")
_SAMPLE = 500
# A string column whose values average this many characters (or words) is free text.
_TEXT_MIN_LENGTH = 30
_TEXT_MIN_WORDS = 4


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """Cheap facts computed once per column and shared by every rule."""

    series: pd.Series
    non_null: pd.Series
    unique: int

    @property
    def is_string(self) -> bool:
        return bool(ptypes.is_object_dtype(self.series) or ptypes.is_string_dtype(self.series))

    @property
    def all_unique(self) -> bool:
        return len(self.non_null) > 0 and self.unique == len(self.non_null)


Rule = Callable[[ColumnFacts], bool]


def _is_boolean(f: ColumnFacts) -> bool:
    if ptypes.is_bool_dtype(f.series):
        return True
    if f.unique == 0 or f.unique > 2:
        return False
    if ptypes.is_numeric_dtype(f.series):
        return set(f.non_null.unique()).issubset({0, 1})
    if f.is_string:
        return {str(v).strip().lower() for v in f.non_null.unique()} <= _BOOL_TOKENS
    return False


def _is_datetime(f: ColumnFacts) -> bool:
    if ptypes.is_datetime64_any_dtype(f.series):
        return True
    if not f.is_string or f.non_null.empty:
        return False
    sample = f.non_null.astype(str).head(_SAMPLE)
    if not sample.str.contains(_DATE_SHAPE).mean() >= 0.95:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.95)


def _is_id_like(f: ColumnFacts) -> bool:
    if not f.all_unique or len(f.non_null) < 20:
        return False
    named_like_id = bool(_ID_NAME.search(str(f.series.name)))
    if ptypes.is_integer_dtype(f.series):
        return named_like_id or bool(f.non_null.is_monotonic_increasing)
    if f.is_string:
        lengths = f.non_null.astype(str).str.len()
        return named_like_id or lengths.mean() < _TEXT_MIN_LENGTH
    return False


def _is_numeric(f: ColumnFacts) -> bool:
    return ptypes.is_numeric_dtype(f.series) and not ptypes.is_bool_dtype(f.series)


def _is_text(f: ColumnFacts) -> bool:
    if not f.is_string or f.non_null.empty:
        return False
    sample = f.non_null.astype(str).head(_SAMPLE)
    words = sample.str.split().str.len()
    return bool(sample.str.len().mean() >= _TEXT_MIN_LENGTH or words.mean() >= _TEXT_MIN_WORDS)


def _is_categorical(_f: ColumnFacts) -> bool:
    return True  # fallback


SEMANTIC_RULES: list[tuple[SemanticType, Rule]] = [
    (SemanticType.BOOLEAN, _is_boolean),
    (SemanticType.DATETIME, _is_datetime),
    (SemanticType.ID_LIKE, _is_id_like),
    (SemanticType.NUMERIC, _is_numeric),
    (SemanticType.TEXT, _is_text),
    (SemanticType.CATEGORICAL, _is_categorical),
]


def column_facts(series: pd.Series) -> ColumnFacts:
    non_null = series.dropna()
    return ColumnFacts(series=series, non_null=non_null, unique=int(non_null.nunique()))


def infer_semantic_type(facts: ColumnFacts) -> SemanticType:
    """Return the first semantic type whose rule matches."""
    return next(kind for kind, rule in SEMANTIC_RULES if rule(facts))
