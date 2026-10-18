import io
import zipfile

import pandas as pd
import pytest

from app.domain.datasets import FileFormat
from app.ingest.errors import InvalidTableError, TableTooLargeError, UnsupportedFormatError
from app.ingest.parquet import read_head, read_parquet, to_parquet_bytes
from app.ingest.readers import TableLimits, read_table
from app.ingest.sniff import sniff_format

LIMITS = TableLimits(max_rows=100, max_columns=10)
FRAME = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})


def _xlsx(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


def _parquet(df: pd.DataFrame) -> bytes:
    return to_parquet_bytes(df)


@pytest.mark.parametrize(
    ("data", "fmt"),
    [
        (b"x,y\n1,a\n", FileFormat.CSV),
        (_parquet(FRAME), FileFormat.PARQUET),
        (_xlsx(FRAME), FileFormat.XLSX),
    ],
)
def test_sniffs_supported_formats(data: bytes, fmt: FileFormat) -> None:
    assert sniff_format(data) is fmt


def _zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "hello")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data",
    [b"", b"   \n", b"\x89PNG\r\n\x1a\nxxxx", b"%PDF-1.7", b"MZ\x90\x00", b"a,b\n\x00\x01", _zip()],
    ids=["empty", "blank", "png", "pdf", "exe", "binary", "zip"],
)
def test_rejects_unsupported_files(data: bytes) -> None:
    with pytest.raises(UnsupportedFormatError):
        sniff_format(data)


@pytest.mark.parametrize("data", [_parquet(FRAME), _xlsx(FRAME), b"x,y\n1,a\n2,b\n3,c\n"])
def test_reads_each_format(data: bytes) -> None:
    df = read_table(data, sniff_format(data), LIMITS)
    assert list(df.columns) == ["x", "y"] and len(df) == 3


def test_csv_delimiter_and_encoding() -> None:
    data = "name;city\nJosé;Málaga\nAnn;Oslo\n".encode("latin-1")
    df = read_table(data, FileFormat.CSV, LIMITS)
    assert df["name"].tolist() == ["José", "Ann"]


@pytest.mark.parametrize(
    ("data", "fmt"),
    [
        (b"x\n" + b"1\n" * 101, FileFormat.CSV),
        (",".join(f"c{i}" for i in range(11)).encode() + b"\n", FileFormat.CSV),
        (_parquet(pd.DataFrame({"x": range(101)})), FileFormat.PARQUET),
    ],
    ids=["csv-rows", "csv-columns", "parquet-rows"],
)
def test_enforces_limits(data: bytes, fmt: FileFormat) -> None:
    with pytest.raises(TableTooLargeError):
        read_table(data, fmt, LIMITS)


@pytest.mark.parametrize("data", [b"x,y\n", b"PAR1garbagePAR1"], ids=["no-rows", "bad-pq"])
def test_rejects_invalid_tables(data: bytes) -> None:
    with pytest.raises(InvalidTableError):
        read_table(data, sniff_format(data), LIMITS)


def test_parquet_round_trip_with_mixed_object_column() -> None:
    df = pd.DataFrame({"mixed": [1, "two", None], "n": [1.0, 2.0, 3.0]})
    back = read_parquet(to_parquet_bytes(df))
    assert back["mixed"].tolist()[:2] == ["1", "two"] and pd.isna(back["mixed"].iloc[2])
    assert read_head(to_parquet_bytes(df), 2).shape == (2, 2)


def test_duplicate_csv_headers_are_made_unique() -> None:
    df = read_table(b"a,a\n1,2\n", FileFormat.CSV, LIMITS)
    assert list(df.columns) == ["a", "a.1"]
