"""DataFrame → DatasetProfile."""

import re

import numpy as np
import pandas as pd

from app.ml.profiling.profile import (
    ColumnProfile,
    CorrelationMatrix,
    DatasetProfile,
    SemanticType,
    TargetCandidate,
)
from app.ml.profiling.semantic import column_facts, infer_semantic_type
from app.ml.profiling.summaries import SUMMARIZERS, finite_or_none
from app.ml.profiling.task import TASK_RULES

# Correlations are O(columns²); beyond this many numeric columns they are skipped.
MAX_CORRELATION_COLUMNS = 60
_TARGET_NAME = re.compile(
    r"^(target|label|y|class|outcome|price|species|diagnosis)$|target|label", re.IGNORECASE
)
_MAX_MISSING_FOR_TARGET = 0.5


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Profile every column, correlate numeric ones, and suggest target columns."""
    n_rows = len(df)
    columns: list[ColumnProfile] = []
    candidates: list[tuple[int, TargetCandidate]] = []

    for position, name in enumerate(df.columns):
        series = df[name]
        facts = column_facts(series)
        semantic = infer_semantic_type(facts)
        missing = int(series.isna().sum())
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                semantic_type=semantic,
                missing=missing,
                missing_pct=_pct(missing, n_rows),
                unique=facts.unique,
                unique_pct=_pct(facts.unique, n_rows - missing),
                **SUMMARIZERS[semantic](facts),
            )
        )
        task = TASK_RULES[semantic](facts)
        if task is not None and n_rows and missing / n_rows <= _MAX_MISSING_FOR_TARGET:
            score = 2 * bool(_TARGET_NAME.search(str(name))) + (position == len(df.columns) - 1)
            reason = _candidate_reason(semantic, facts.unique, task)
            candidates.append(
                (score, TargetCandidate(column=str(name), task_type=task, reason=reason))
            )

    candidates.sort(key=lambda item: -item[0])  # stable: ties keep column order
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=df.shape[1],
        duplicate_rows=int(df.duplicated().sum()) if n_rows else 0,
        columns=columns,
        correlations=_correlations(df, columns),
        target_candidates=[candidate for _, candidate in candidates],
    )


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def _candidate_reason(semantic: SemanticType, unique: int, task: str) -> str:
    return f"{unique} distinct {semantic} values → {task}."


def _correlations(df: pd.DataFrame, columns: list[ColumnProfile]) -> CorrelationMatrix | None:
    numeric = [c.name for c in columns if c.semantic_type is SemanticType.NUMERIC]
    if len(numeric) < 2 or len(numeric) > MAX_CORRELATION_COLUMNS:
        return None
    frame = df[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matrix = frame.corr(method="pearson")
    values = [[finite_or_none(v) for v in row] for row in matrix.to_numpy()]
    return CorrelationMatrix(columns=numeric, values=values)
