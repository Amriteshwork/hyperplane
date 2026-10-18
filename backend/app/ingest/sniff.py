"""Detect the real file format from its bytes, never from the file name."""

import io
import zipfile

from app.domain.datasets import FileFormat
from app.ingest.errors import UnsupportedFormatError

# Formats we refuse outright, with a helpful name. Checked before the CSV fallback.
_KNOWN_BINARY = {
    b"\x89PNG": "PNG image",
    b"\xff\xd8\xff": "JPEG image",
    b"GIF8": "GIF image",
    b"%PDF": "PDF document",
    b"\x1f\x8b": "gzip archive",
    b"\xd0\xcf\x11\xe0": "legacy Excel (.xls) or other OLE file — save it as .xlsx or .csv",
    b"\x7fELF": "executable",
    b"MZ": "executable",
}
_CSV_PROBE_BYTES = 64 * 1024


def sniff_format(data: bytes) -> FileFormat:
    """Return the format of `data` or raise `UnsupportedFormatError`."""
    if not data.strip():
        raise UnsupportedFormatError("The file is empty.")
    if data[:4] == b"PAR1" and data[-4:] == b"PAR1":
        return FileFormat.PARQUET
    if data[:4] == b"PK\x03\x04":
        return _sniff_zip(data)
    for magic, description in _KNOWN_BINARY.items():
        if data.startswith(magic):
            raise UnsupportedFormatError(f"Unsupported file type: {description}.")
    if b"\x00" in data[:_CSV_PROBE_BYTES]:
        raise UnsupportedFormatError("Unsupported binary file. Upload CSV, Parquet or XLSX.")
    return FileFormat.CSV


def _sniff_zip(data: bytes) -> FileFormat:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        raise UnsupportedFormatError("Corrupt ZIP/XLSX file.") from None
    if "xl/workbook.xml" in names:
        return FileFormat.XLSX
    raise UnsupportedFormatError("ZIP archives are not supported. Upload CSV, Parquet or XLSX.")
