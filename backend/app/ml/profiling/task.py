"""Task-type inference for a chosen target column: `TASK_RULES[semantic_type]`."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from app.ml.profiling.profile import SemanticType
from app.ml.profiling.semantic import ColumnFacts, column_facts, infer_semantic_type
from app.ml.types import TaskType

# A target with more classes than this is not treated as classification.
MAX_CLASSES = 50
# Integer-valued numeric targets with at most this many distinct values look like class labels.
MAX_INTEGER_CLASSES = 20

TaskRule = Callable[[ColumnFacts], TaskType | None]


def _classes(f: ColumnFacts) -> TaskType | None:
    if f.unique < 2 or f.unique > MAX_CLASSES:
        return None
    return TaskType.BINARY if f.unique == 2 else TaskType.MULTICLASS


def _numeric(f: ColumnFacts) -> TaskType | None:
    if f.unique < 2:
        return None
    values = pd.to_numeric(f.non_null, errors="coerce").dropna().to_numpy(dtype=float)
    integer_valued = bool(np.all(np.isfinite(values)) and np.all(values == np.round(values)))
    if integer_valued and f.unique <= MAX_INTEGER_CLASSES:
        return _classes(f)
    return TaskType.REGRESSION


def _not_a_target(_f: ColumnFacts) -> TaskType | None:
    return None


TASK_RULES: dict[SemanticType, TaskRule] = {
    SemanticType.NUMERIC: _numeric,
    SemanticType.BOOLEAN: _classes,
    SemanticType.CATEGORICAL: _classes,
    SemanticType.DATETIME: _not_a_target,
    SemanticType.TEXT: _not_a_target,
    SemanticType.ID_LIKE: _not_a_target,
}


def infer_task_type(series: pd.Series) -> TaskType | None:
    """Task implied by using `series` as the target, or None if it can't be one."""
    facts = column_facts(series)
    return TASK_RULES[infer_semantic_type(facts)](facts)
