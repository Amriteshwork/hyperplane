"""DataFrame ⇄ Parquet bytes (the canonical stored form of every dataset)."""

import io

import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import infer_dtype


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise `df`; object columns with mixed types are stored as strings."""
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == object and infer_dtype(df[column], skipna=True).startswith("mixed"):
            df[column] = df[column].where(df[column].isna(), df[column].astype(str))
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def read_parquet(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")


def read_head(data: bytes, rows: int) -> pd.DataFrame:
    """First `rows` rows without decoding the whole file."""
    parquet = pq.ParquetFile(io.BytesIO(data))
    batches = parquet.iter_batches(batch_size=max(rows, 1))
    first = next(batches, None)
    if first is None:
        return parquet.schema_arrow.empty_table().to_pandas()
    return first.to_pandas().head(rows)
