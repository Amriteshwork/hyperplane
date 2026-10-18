"""Repository protocols services depend on.

Methods are added with each aggregate's table (M0 "Upload → Parquet" and
"Worker + job pipeline"); until then they only fix the names used in wiring.
"""

from typing import Protocol


class DatasetRepository(Protocol):
    """Persistence for datasets."""


class RunRepository(Protocol):
    """Persistence for experiments and their runs."""
