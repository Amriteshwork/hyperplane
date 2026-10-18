"""`DatasetProfile`: everything the profiler reports about a table. JSON-safe (NaN → None)."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.ml.types import TaskType


class SemanticType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id-like"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Histogram(_Frozen):
    """`counts[i]` values fall in `[edges[i], edges[i + 1])` (last bin closed)."""

    edges: list[float]
    counts: list[int]


class NumericSummary(_Frozen):
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    q25: float | None
    median: float | None
    q75: float | None
    skew: float | None
    zeros: int
    histogram: Histogram | None


class ValueCount(_Frozen):
    value: str
    count: int


class CategoricalSummary(_Frozen):
    """Most frequent values (up to 20); the rest are counted in `other`."""

    top: list[ValueCount]
    other: int


class DatetimeSummary(_Frozen):
    min: str | None
    max: str | None


class TextSummary(_Frozen):
    mean_length: float
    max_length: int


class ColumnProfile(_Frozen):
    name: str
    dtype: str = Field(description="pandas dtype")
    semantic_type: SemanticType
    missing: int
    missing_pct: float
    unique: int = Field(description="Distinct non-missing values (cardinality).")
    unique_pct: float
    numeric: NumericSummary | None = None
    categorical: CategoricalSummary | None = None
    datetime: DatetimeSummary | None = None
    text: TextSummary | None = None


class CorrelationMatrix(_Frozen):
    """Pearson correlation between numeric columns; `values[i][j]` pairs `columns[i]`, `[j]`."""

    columns: list[str]
    values: list[list[float | None]]


class TargetCandidate(_Frozen):
    column: str
    task_type: TaskType
    reason: str


class DatasetProfile(_Frozen):
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    columns: list[ColumnProfile]
    correlations: CorrelationMatrix | None = None
    target_candidates: list[TargetCandidate] = []
