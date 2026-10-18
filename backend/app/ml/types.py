"""Types shared across ml/ sub-packages (kept here to avoid import cycles)."""

from enum import StrEnum


class TaskType(StrEnum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"
