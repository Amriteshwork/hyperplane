"""DataFrame → DatasetProfile, plus task-type inference for a chosen target."""

from app.ml.profiling.profile import DatasetProfile, SemanticType
from app.ml.profiling.profiler import profile_dataframe
from app.ml.profiling.task import infer_task_type

__all__ = ["DatasetProfile", "SemanticType", "infer_task_type", "profile_dataframe"]
