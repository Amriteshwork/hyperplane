"""Per-semantic-type column summaries: `SUMMARIZERS[semantic_type]` fills one profile section."""

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from app.ml.profiling.profile import (
    CategoricalSummary,
    DatetimeSummary,
    Histogram,
    NumericSummary,
    SemanticType,
    TextSummary,
    ValueCount,
)
from app.ml.profiling.semantic import ColumnFacts

HISTOGRAM_BINS = 20
TOP_VALUES = 20

Summary = dict[str, Any]
Summarizer = Callable[[ColumnFacts], Summary]


def finite_or_none(value: object) -> float | None:
    """JSON-safe float: NaN/inf and non-numbers become None."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _numeric(f: ColumnFacts) -> Summary:
    values = pd.to_numeric(f.non_null, errors="coerce").astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        empty = dict.fromkeys(("min", "max", "mean", "std", "q25", "median", "q75", "skew"))
        return {"numeric": NumericSummary(**empty, zeros=0, histogram=None)}
    counts, edges = np.histogram(values, bins=min(HISTOGRAM_BINS, max(1, values.nunique())))
    q25, median, q75 = values.quantile([0.25, 0.5, 0.75])
    summary = NumericSummary(
        min=finite_or_none(values.min()),
        max=finite_or_none(values.max()),
        mean=finite_or_none(values.mean()),
        std=finite_or_none(values.std()),
        q25=finite_or_none(q25),
        median=finite_or_none(median),
        q75=finite_or_none(q75),
        skew=finite_or_none(values.skew()),
        zeros=int((values == 0).sum()),
        histogram=Histogram(edges=[float(e) for e in edges], counts=[int(c) for c in counts]),
    )
    return {"numeric": summary}


def _categorical(f: ColumnFacts) -> Summary:
    counts = f.non_null.astype(str).value_counts()
    top = [ValueCount(value=str(v), count=int(c)) for v, c in counts.head(TOP_VALUES).items()]
    other = int(counts.iloc[TOP_VALUES:].sum())
    return {"categorical": CategoricalSummary(top=top, other=other)}


def _datetime(f: ColumnFacts) -> Summary:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(f.non_null, errors="coerce", format="mixed").dropna()
    if parsed.empty:
        return {"datetime": DatetimeSummary(min=None, max=None)}
    return {"datetime": DatetimeSummary(min=parsed.min().isoformat(), max=parsed.max().isoformat())}


def _text(f: ColumnFacts) -> Summary:
    lengths = f.non_null.astype(str).str.len()
    return {
        "text": TextSummary(
            mean_length=round(float(lengths.mean()), 1) if len(lengths) else 0.0,
            max_length=int(lengths.max()) if len(lengths) else 0,
        )
    }


def _none(_f: ColumnFacts) -> Summary:
    return {}


SUMMARIZERS: dict[SemanticType, Summarizer] = {
    SemanticType.NUMERIC: _numeric,
    SemanticType.CATEGORICAL: _categorical,
    SemanticType.BOOLEAN: _categorical,
    SemanticType.DATETIME: _datetime,
    SemanticType.TEXT: _text,
    SemanticType.ID_LIKE: _none,
}
