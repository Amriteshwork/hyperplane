"""Read-only, validated index of `content/`. Loading fails loudly on malformed files."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.content.models import (
    AlgorithmContent,
    AlgorithmFrontmatter,
    Concept,
    ConceptFrontmatter,
    Family,
    FamilyFrontmatter,
    Resource,
)

_FRONTMATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n?(?P<body>.*)\Z", re.DOTALL)
_RESOURCES = TypeAdapter(tuple[Resource, ...])


class ContentError(ValueError):
    """A content file is missing, malformed, or fails its schema."""


class UnknownContentError(KeyError):
    """No family/algorithm/concept with this id."""


class ContentCatalog:
    def __init__(
        self,
        families: Sequence[Family] = (),
        algorithms: Sequence[AlgorithmContent] = (),
        concepts: Sequence[Concept] = (),
    ) -> None:
        self._families = {f.id: f for f in sorted(families, key=lambda f: f.meta.order)}
        self._algorithms = {a.id: a for a in algorithms}
        self._concepts = {c.id: c for c in concepts}

    @classmethod
    def load(cls, root: Path) -> ContentCatalog:
        """Index `root`; a missing folder gives an empty catalog (like `SampleCatalog`)."""
        if not root.is_dir():
            return cls()
        families = [
            Family(id=path.stem, **_parts(path, FamilyFrontmatter))
            for path in sorted((root / "families").glob("*.md"))
            if path.name != "README.md"
        ]
        algorithms = [
            AlgorithmContent(
                id=folder.name,
                **_parts(folder / "overview.md", AlgorithmFrontmatter),
                resources=_resources(folder / "resources.yaml"),
            )
            for folder in _folders(root / "algorithms")
        ]
        concepts = [
            Concept(
                id=folder.name,
                **_parts(folder / "concept.md", ConceptFrontmatter),
                resources=_resources(folder / "resources.yaml"),
            )
            for folder in _folders(root / "concepts")
        ]
        return cls(families, algorithms, concepts)

    def families(self) -> Sequence[Family]:
        return list(self._families.values())

    def algorithm(self, algorithm_id: str) -> AlgorithmContent:
        return _get(self._algorithms, algorithm_id)

    def concept(self, concept_id: str) -> Concept:
        return _get(self._concepts, concept_id)

    def concepts(self) -> Sequence[Concept]:
        return list(self._concepts.values())

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    def family_of(self, algorithm_id: str) -> Family | None:
        for family in self._families.values():
            if any(a.id == algorithm_id for a in family.meta.algorithms):
                return family
        return None


def _get[T](index: dict[str, T], key: str) -> T:
    try:
        return index[key]
    except KeyError:
        raise UnknownContentError(key) from None


def _folders(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


class _Parts[M: BaseModel](TypedDict):
    meta: M
    body: str


def _parts[M: BaseModel](path: Path, meta_model: type[M]) -> _Parts[M]:
    """Frontmatter and body of a Markdown file."""
    match = _FRONTMATTER.match(_read(path))
    if not match:
        raise ContentError(f"{path}: missing '---' YAML frontmatter")
    try:
        meta = meta_model.model_validate(yaml.safe_load(match["meta"]) or {})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid frontmatter: {exc}") from exc
    return {"meta": meta, "body": match["body"].strip() + "\n"}


def _resources(path: Path) -> tuple[Resource, ...]:
    try:
        return _RESOURCES.validate_python(yaml.safe_load(_read(path)) or [])
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid resources: {exc}") from exc


def _read(path: Path) -> str:
    try:
        return path.read_text("utf-8")
    except FileNotFoundError:
        raise ContentError(f"{path}: missing required file") from None
