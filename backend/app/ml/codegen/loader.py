"""Reads a pack's templates and `annotations.yaml` (one section per template name)."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from app.ml.base import Annotation, ExplainedTemplate

ANNOTATIONS_FILE = "annotations.yaml"
_ANNOTATIONS = TypeAdapter(dict[str, dict[str, Annotation]])


@dataclass(frozen=True, slots=True)
class TemplateFile:
    file: str
    title: str


def load_templates(pack_dir: Path, files: dict[str, TemplateFile]) -> dict[str, ExplainedTemplate]:
    """`files` maps template name → its file in `pack_dir` and display title."""
    raw = yaml.safe_load((pack_dir / ANNOTATIONS_FILE).read_text("utf-8")) or {}
    annotations = _ANNOTATIONS.validate_python(raw)
    unknown = sorted(set(annotations) - set(files))
    if unknown:
        raise ValueError(f"{pack_dir / ANNOTATIONS_FILE}: sections for unknown templates {unknown}")
    templates = {}
    for name, entry in files.items():
        templated = entry.file.endswith(".j2")
        templates[name] = ExplainedTemplate(
            source=(pack_dir / entry.file).read_text("utf-8"),
            annotations=annotations.get(name, {}),
            filename=entry.file.removesuffix(".j2"),
            title=entry.title,
            templated=templated,
        )
    return templates
