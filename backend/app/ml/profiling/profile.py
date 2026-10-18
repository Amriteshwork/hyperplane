"""`DatasetProfile`: what the profiler knows about a dataset. Fields grow with M0 profiling."""

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
