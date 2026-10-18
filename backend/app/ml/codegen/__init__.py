"""Templates → displayed code: render Jinja, strip `# @a:` anchors, resolve their line spans."""

from app.ml.codegen.anchors import (
    AnchorMismatchError,
    ExplainedSpan,
    RenderedCode,
    find_anchors,
    strip_anchors,
)
from app.ml.codegen.loader import TemplateFile, load_templates
from app.ml.codegen.render import CodegenContext, render

__all__ = [
    "AnchorMismatchError",
    "CodegenContext",
    "ExplainedSpan",
    "RenderedCode",
    "TemplateFile",
    "find_anchors",
    "load_templates",
    "render",
    "strip_anchors",
]
