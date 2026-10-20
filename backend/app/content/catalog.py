"""Read-only, validated index of `content/`. Loading fails loudly on malformed files."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.content.code_imports import code_uses
from app.content.graph import ConceptGraphError, check_graph, learning_path, reading_order
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
    ResourceLink,
    code_module,
)
from app.ml.base import Annotation
from app.ml.codegen import AnchorMismatchError, strip_anchors

_FRONTMATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n?(?P<body>.*)\Z", re.DOTALL)


class _ResourceRef(BaseModel):
    """One entry of a page's `resources.yaml`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    note: str = Field(min_length=1)


_RESOURCE_REFS = TypeAdapter(tuple[_ResourceRef, ...])
_CODE_ANNOTATIONS = TypeAdapter(dict[CodeLibrary, dict[str, Annotation]])
CODE_ANNOTATIONS_FILE = "annotations.yaml"
_IGNORED = {".DS_Store", "__pycache__"}


class ContentError(ValueError):
    """A content file is missing, malformed, or fails its schema."""


class UnknownContentError(KeyError):
    """No family/algorithm/concept (or concept code in that library) with this id."""


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
        return cls(families, algorithms, _link_concepts(concepts), list(resources.values()))

    def families(self) -> Sequence[Family]:
        return list(self._families.values())

    def algorithms(self) -> Sequence[AlgorithmContent]:
        return list(self._algorithms.values())

    def curriculum(self) -> Sequence[str]:
        """Algorithm ids in teaching order: family by family, as each family page lists them."""
        return [a.id for family in self._families.values() for a in family.meta.algorithms]

    def algorithm(self, algorithm_id: str) -> AlgorithmContent:
        return _get(self._algorithms, algorithm_id)

    def concept(self, concept_id: str) -> Concept:
        return _get(self._concepts, concept_id)

    def concepts(self) -> Sequence[Concept]:
        return list(self._concepts.values())

    def resources(self) -> Sequence[Resource]:
        return list(self._resources.values())

    def learning_path(self, concept_id: str) -> Sequence[Concept]:
        """Every prerequisite of the concept, transitively, each after everything it needs."""
        _get(self._concepts, concept_id)
        prerequisites = {c.id: c.meta.prerequisites for c in self._concepts.values()}
        return [self._concepts[i] for i in learning_path(prerequisites, concept_id)]

    def code_with_dependencies(
        self, concept_id: str, library: CodeLibrary
    ) -> Sequence[tuple[str, ConceptCode]]:
        """The concept's code in `library`, then every file it imports, transitively, once each."""
        found: dict[str, ConceptCode] = {}

        def visit(concept: str) -> None:
            code = next(
                (c for c in _get(self._concepts, concept).code if c.library is library), None
            )
            if code is None:
                raise UnknownContentError(f"concepts/{concept} has no {library} code")
            found[concept] = code
            for use in code.uses:
                if use.concept not in found:
                    visit(use.concept)

        visit(concept_id)
        return list(found.items())

    def concept_index(self) -> Sequence[Concept]:
        """Every concept, each after everything it needs."""
        return self.reading_order(sorted(self._concepts))

    def reading_order(self, concept_ids: Sequence[str]) -> Sequence[Concept]:
        """The concepts plus everything they build on, each after what it needs."""
        for concept_id in concept_ids:
            _get(self._concepts, concept_id)
        prerequisites = {c.id: c.meta.prerequisites for c in self._concepts.values()}
        return [self._concepts[i] for i in reading_order(prerequisites, concept_ids)]

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


def _resource_refs(path: Path, index: dict[str, Resource]) -> tuple[ResourceLink, ...]:
    """A page's `resources.yaml`: `{id, note}` entries in display order (ADR 0016)."""
    try:
        refs = _RESOURCE_REFS.validate_python(yaml.safe_load(_read(path)) or [])
    except (yaml.YAMLError, ValidationError) as exc:
        raise ContentError(f"{path}: invalid resources: {exc}") from exc
    ids = [r.id for r in refs]
    if unknown := [i for i in ids if i not in index]:
        raise ContentError(f"{path}: unknown resources {unknown}")
    if len(set(ids)) != len(ids):
        raise ContentError(f"{path}: duplicate resource ids")
    return tuple(ResourceLink(resource=index[r.id], note=r.note.strip()) for r in refs)


def _link_concepts(concepts: list[Concept]) -> list[Concept]:
    """Check the prerequisite graph, then resolve what each concept's code imports."""
    prerequisites = {c.id: c.meta.prerequisites for c in concepts}
    sources = {c.id: {code.library: code.source for code in c.code} for c in concepts}
    try:
        check_graph(prerequisites)
        linked = []
        for concept in concepts:
            ancestors = set(learning_path(prerequisites, concept.id))
            code = tuple(_with_uses(concept.id, c, ancestors, sources) for c in concept.code)
            linked.append(concept.model_copy(update={"code": code}))
        return linked
    except ConceptGraphError as exc:
        raise ContentError(str(exc)) from exc


def _with_uses(
    concept_id: str,
    code: ConceptCode,
    ancestors: set[str],
    sources: dict[str, dict[CodeLibrary, str]],
) -> ConceptCode:
    filename = f"{code_module(concept_id, code.library)}.py"
    uses = code_uses(filename, code.source, code.library, ancestors, sources)
    return code.model_copy(update={"uses": uses})


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
