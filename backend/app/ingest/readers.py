"""`READERS[format]`: bytes → DataFrame, enforcing row/column limits while reading."""

import csv
import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq

from app.domain.datasets import FileFormat
from app.ingest.errors import InvalidTableError, TableTooLargeError

_DELIMITERS = ",;\t|"
# An XLSX inflating to more than this many times its size is treated as a zip bomb.
_MAX_XLSX_INFLATION = 100


@dataclass(frozen=True, slots=True)
class TableLimits:
    max_rows: int
    max_columns: int


Reader = Callable[[bytes, TableLimits], pd.DataFrame]


def _read_csv(data: bytes, limits: TableLimits) -> pd.DataFrame:
    encoding = "utf-8-sig" if _is_utf8(data) else "latin-1"
    text_head = data[: 64 * 1024].decode(encoding, errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(text_head, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        delimiter = ","
    try:
        return pd.read_csv(
            io.BytesIO(data), sep=delimiter, encoding=encoding, nrows=limits.max_rows + 1
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InvalidTableError(f"Could not parse the CSV: {_first_line(exc)}") from None


def _read_parquet(data: bytes, limits: TableLimits) -> pd.DataFrame:
    try:
        parquet = pq.ParquetFile(io.BytesIO(data))
    except Exception:  # noqa: BLE001  (pyarrow raises several types for corrupt files)
        raise InvalidTableError("Could not read the Parquet file.") from None
    _check_limits(parquet.metadata.num_rows, parquet.metadata.num_columns, limits)
    return parquet.read().to_pandas()


def _read_xlsx(data: bytes, limits: TableLimits) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        inflated = sum(info.file_size for info in archive.infolist())
    if inflated > _MAX_XLSX_INFLATION * len(data):
        raise InvalidTableError("The XLSX file expands to an implausible size.")
    try:
        # First sheet only; the Lab works on one table.
        return pd.read_excel(io.BytesIO(data), sheet_name=0, nrows=limits.max_rows + 1)
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise InvalidTableError(f"Could not read the XLSX file: {_first_line(exc)}") from None


READERS: dict[FileFormat, Reader] = {
    FileFormat.CSV: _read_csv,
    FileFormat.PARQUET: _read_parquet,
    FileFormat.XLSX: _read_xlsx,
}


def read_table(data: bytes, fmt: FileFormat, limits: TableLimits) -> pd.DataFrame:
    """Parse `data` as `fmt` and validate shape; column names become strings."""
    df = READERS[fmt](data, limits)
    _check_limits(len(df), df.shape[1], limits)
    if df.shape[1] == 0 or len(df) == 0:
        raise InvalidTableError("The table has no rows or no columns.")
    df.columns = [str(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise InvalidTableError("Column names must be unique.")
    return df


def _check_limits(rows: int, columns: int, limits: TableLimits) -> None:
    if rows > limits.max_rows:
        raise TableTooLargeError(f"Too many rows: the limit is {limits.max_rows:,}.")
    if columns > limits.max_columns:
        raise TableTooLargeError(f"Too many columns: the limit is {limits.max_columns:,}.")


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
