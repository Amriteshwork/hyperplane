class IngestError(Exception):
    """Base for problems with an uploaded file; the message is safe to show to the user."""


class UnsupportedFormatError(IngestError):
    """Not CSV, Parquet or XLSX."""


class InvalidTableError(IngestError):
    """The file parses, but not into a usable table."""


class TableTooLargeError(IngestError):
    """Too many rows or columns."""
