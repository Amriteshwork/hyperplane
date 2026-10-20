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
    CodeLibrary,
    Concept,
    ConceptCode,
    ConceptFrontmatter,
    Family,
    FamilyFrontmatter,
    Resource,
)
from app.ml.base import Annotation
from app.ml.codegen import AnchorMismatchError, strip_anchors

_FRONTMATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n?(?P<body>.*)\Z", re.DOTALL)
_RESOURCE_IDS = TypeAdapter(tuple[str, ...])
_CODE_ANNOTATIONS = TypeAdapter(dict[CodeLibrary, dict[str, Annotation]])
CODE_ANNOTATIONS_FILE = "annotations.yaml"
_IGNORED = {".DS_Store", "__pycache__"}


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
        resources: Sequence[Resource] = (),
    ) -> None:
        self._families = {f.id: f for f in sorted(families, key=lambda f: f.meta.order)}
        self._algorithms = {a.id: a for a in algorithms}
        self._concepts = {c.id: c for c in concepts}
        self._resources = {r.id: r for r in resources}

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
        resources = {
            path.stem: _resource(path) for path in sorted((root / "resources").glob("*.yaml"))
        }
        algorithms = [
            AlgorithmContent(
                id=folder.name,
                **_parts(folder / "overview.md", AlgorithmFrontmatter),
                resources=_resource_refs(folder / "resources.yaml", resources),
            )
            for folder in _folders(root / "algorithms")
        ]
        concepts = [
            Concept(
                id=folder.name,
                **_parts(folder / "concept.md", ConceptFrontmatter),
                resources=_resource_refs(folder / "resources.yaml", resources),
                code=_concept_code(folder / "code"),
            )
            for folder in _folders(root / "concepts")
        ]
        return cls(families, algorithms, concepts, list(resources.values()))

    def families(self) -> Sequence[Family]:
        return list(self._families.values())

    def algorithms(self) -> Sequence[AlgorithmContent]:
        return list(self._algorithms.values())

    def algorithm(self, algorithm_id: str) -> AlgorithmContent:
        return _get(self._algorithms, algorithm_id)

    def concept(self, concept_id: str) -> Concept:
        return _get(self._concepts, concept_id)

    def concepts(self) -> Sequence[Concept]:
        return list(self._concepts.values())

    def resources(self) -> Sequence[Resource]:
        return list(self._resources.values())

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


def _resource(path: Path) -> Resource:
    """A resource file; its id is the filename, so it can't drift from references."""
    try:
        data = yaml.safe_load(_read(path))
        if not isinstance(data, dict):
            raise ContentError(f"{path}: expected a YAML mapping")
        return Resource.model_validate({**data, "id": path.stem})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid resource: {exc}") from exc


def _resource_refs(path: Path, index: dict[str, Resource]) -> tuple[Resource, ...]:
    """An algorithm's or concept's `resources.yaml`: a list of resource ids, in display order."""
    try:
        ids = _RESOURCE_IDS.validate_python(yaml.safe_load(_read(path)) or [])
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid resources: {exc}") from exc
    if unknown := [i for i in ids if i not in index]:
        raise ContentError(f"{path}: unknown resources {unknown}")
    if len(set(ids)) != len(ids):
        raise ContentError(f"{path}: duplicate resource ids")
    return tuple(index[i] for i in ids)


def _concept_code(folder: Path) -> tuple[ConceptCode, ...]:
    """A concept's optional `code/<library>.py` files, in `CodeLibrary` order.

    `code/annotations.yaml` has one section per library; anchors and entries must match 1:1.
    """
    if not folder.is_dir():
        return ()
    annotations = _code_annotations(folder / CODE_ANNOTATIONS_FILE)
    libraries = {library.value: library for library in CodeLibrary}
    code: dict[CodeLibrary, ConceptCode] = {}
    for path in sorted(p for p in folder.iterdir() if p.name not in _IGNORED):
        if path.name == CODE_ANNOTATIONS_FILE:
            continue
        if path.suffix != ".py" or path.stem not in libraries:
            raise ContentError(f"{path}: expected one of {[f'{name}.py' for name in libraries]}")
        library = libraries[path.stem]
        code[library] = ConceptCode(
            library=library, source=_read(path), annotations=annotations.get(library, {})
        )
        try:
            strip_anchors(code[library].source, code[library].annotations, filename=str(path))
        except AnchorMismatchError as exc:
            raise ContentError(str(exc)) from exc
    if orphaned := sorted(set(annotations) - set(code)):
        raise ContentError(f"{folder / CODE_ANNOTATIONS_FILE}: sections without code {orphaned}")
    return tuple(code[library] for library in CodeLibrary if library in code)


def _code_annotations(path: Path) -> dict[CodeLibrary, dict[str, Annotation]]:
    if not path.is_file():
        return {}
    try:
        return _CODE_ANNOTATIONS.validate_python(yaml.safe_load(_read(path)) or {})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid annotations: {exc}") from exc


def _read(path: Path) -> str:
    try:
        return path.read_text("utf-8")
    except FileNotFoundError:
        raise ContentError(f"{path}: missing required file") from None
