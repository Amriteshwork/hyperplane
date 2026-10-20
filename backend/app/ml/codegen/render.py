"""Jinja rendering of a pack's `train.py.j2` for one configuration."""

from typing import Any

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.ml.base import ExplainedTemplate
from app.ml.codegen.anchors import RenderedCode, strip_anchors
from app.ml.types import TaskType


class CodegenContext(BaseModel):
    """Everything a training template may reference. User-supplied values reach the code only
    through the `py` filter, which emits Python literals (never raw text)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_file: str
    target: str
    #: Lets one template print the metrics for each task type it supports.
    task_type: TaskType
    numeric_features: list[str]
    categorical_features: list[str] = []
    test_size: float = Field(gt=0, lt=1)
    seed: int
    params: dict[str, JsonValue]


def _py(value: Any) -> str:
    """A Python literal for JSON-like data; `repr` quotes and escapes strings safely."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_py(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_py(k)}: {_py(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Cannot render {type(value).__name__} as a Python literal")


_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,  # output is Python source, not HTML; values go through `py`
)
_ENV.filters["py"] = _py


def render(template: ExplainedTemplate, context: CodegenContext | None = None) -> RenderedCode:
    """Displayed code + spans. `context` is required for Jinja templates and ignored otherwise."""
    source = template.source
    if template.templated:
        if context is None:
            raise ValueError(f"{template.filename} is a template and needs a CodegenContext")
        source = _ENV.from_string(source).render(**context.model_dump())
    return strip_anchors(source, template.annotations, filename=template.filename)
