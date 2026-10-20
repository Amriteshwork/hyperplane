"""Content queries: the learning catalog joined with the registered packs."""

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass

from app.content.catalog import ContentCatalog, UnknownContentError
from app.content.models import (
    CODE_LIBRARY_PACKAGES,
    CODE_LIBRARY_TITLES,
    CONCEPT_AREA_TITLES,
    CodeLibrary,
    Concept,
    ConceptCode,
    Family,
    ResourceLink,
    code_module,
)
from app.core.errors import NotFoundError
from app.ml.base import AlgorithmPack
from app.ml.codegen import CodegenContext, RenderedCode, render, strip_anchors
from app.ml.registry import AlgorithmRegistry, UnknownAlgorithmError
from app.schemas.content import (
    AlgorithmOut,
    AlgorithmRefOut,
    CodeAnnotationOut,
    CodeOut,
    CodeVariantOut,
    ConceptAreaOut,
    ConceptCodeOut,
    ConceptCodeUseOut,
    ConceptOut,
    ConceptRefOut,
    ConceptSummaryOut,
    FamilyAlgorithmOut,
    FamilyOut,
    ResourceOut,
)

#: Placeholder data for showing a training template outside a run.
_EXAMPLE = {
    "data_file": "data.csv",
    "target": "target",
    "numeric_features": ["feature_1", "feature_2", "feature_3"],
    "categorical_features": [],
    "test_size": 0.2,
    "seed": 42,
}


#: Fixed timestamp so the same content always zips to the same bytes.
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class CodeBundle:
    """A zip of concept code with everything it imports, ready to unzip and run."""

    filename: str
    content: bytes


class ContentService:
    def __init__(self, catalog: ContentCatalog, registry: AlgorithmRegistry) -> None:
        self._catalog = catalog
        self._registry = registry

    def list_families(self) -> Sequence[FamilyOut]:
        return [self._family(f) for f in self._catalog.families()]

    def get_algorithm(self, algorithm_id: str) -> AlgorithmOut:
        pack = self._pack(algorithm_id)
        content = self._catalog.algorithm(algorithm_id)
        family = self._catalog.family_of(algorithm_id)
        return AlgorithmOut(
            id=pack.spec.id,
            name=pack.spec.name,
            summary=content.meta.summary,
            family=self._family(family) if family else None,
            task_types=sorted(pack.spec.task_types),
            difficulty=pack.spec.difficulty,
            overview=content.body,
            concepts=self._concept_refs(pack.spec.concepts),
            learning_path=self._concept_refs(
                [c.id for c in self._catalog.reading_order(pack.spec.concepts)]
            ),
            resources=_resources(content.resources),
            params_schema=pack.params_model.model_json_schema(),
            code_variants=[
                CodeVariantOut(id=name, title=t.title, filename=t.filename)
                for name, t in pack.templates().items()
            ],
        )

    def get_example_code(self, algorithm_id: str, variant: str) -> CodeOut:
        """A pack's template rendered with placeholder settings and default parameters."""
        pack = self._pack(algorithm_id)
        template = pack.templates().get(variant)
        if template is None:
            raise NotFoundError(f"{algorithm_id!r} has no code variant {variant!r}.")
        context = CodegenContext.model_validate(
            {
                **_EXAMPLE,
                # Templates may branch on the task; show the first one the pack supports.
                "task_type": min(pack.spec.task_types),
                "params": pack.params_model().model_dump(mode="json"),
            }
        )
        return self._code(variant, template.title, render(template, context), "example")

    def list_concept_areas(self) -> Sequence[ConceptAreaOut]:
        """Every concept by area (areas in enum order), each after what it needs."""
        index = self._catalog.concept_index()
        return [
            ConceptAreaOut(
                id=area,
                title=CONCEPT_AREA_TITLES[area],
                concepts=[
                    ConceptSummaryOut(
                        id=c.id,
                        title=c.meta.title,
                        summary=c.meta.summary,
                        algorithms=self._algorithms_using(c.id),
                    )
                    for c in index
                    if c.meta.area is area
                ],
            )
            for area in CONCEPT_AREA_TITLES
            if any(c.meta.area is area for c in index)
        ]

    def get_concept(self, concept_id: str) -> ConceptOut:
        concept = self._concept(concept_id)
        return ConceptOut(
            id=concept.id,
            title=concept.meta.title,
            summary=concept.meta.summary,
            body=concept.body,
            prerequisites=self._concept_refs(concept.meta.prerequisites),
            learning_path=self._concept_refs(
                [c.id for c in self._catalog.learning_path(concept.id)]
            ),
            resources=_resources(concept.resources),
            algorithms=self._algorithms_using(concept_id),
            code=[self._concept_code(concept.id, c) for c in concept.code],
        )

    def get_concept_code_bundle(self, concept_id: str, library: CodeLibrary) -> CodeBundle:
        """The code shown on the page plus every file it imports, in one folder (ADR 0016)."""
        try:
            files = self._catalog.code_with_dependencies(concept_id, library)
        except UnknownContentError:
            raise NotFoundError(f"No {library} code for concept {concept_id!r}.") from None
        shown = [(self._concept(i), self._concept_code(i, code)) for i, code in files]
        (concept, main), *dependencies = shown
        readme = "\n".join(
            [
                f"# {concept.meta.title} ({main.title})",
                "",
                f"From Hyperplane: /concepts/{concept.id}",
                "",
                f"Run it from this folder (needs `pip install {CODE_LIBRARY_PACKAGES[library]}`):",
                "",
                f"    python {main.filename}",
                "",
                "Files:",
                "",
                f"- {main.filename}: {concept.meta.title}",
                *(f"- {f.filename}: {c.meta.title}, which it builds on" for c, f in dependencies),
                "",
            ]
        )
        folder = code_module(concept_id, library)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in [("README.md", readme), *((f.filename, f.code) for _, f in shown)]:
                info = zipfile.ZipInfo(f"{folder}/{name}", date_time=_ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, text)
        return CodeBundle(filename=f"{folder}.zip", content=buffer.getvalue())

    def _concept_code(self, concept_id: str, code: ConceptCode) -> ConceptCodeOut:
        filename = f"{code_module(concept_id, code.library)}.py"
        rendered = strip_anchors(code.source, code.annotations, filename=filename)
        return ConceptCodeOut(
            library=code.library,
            title=CODE_LIBRARY_TITLES[code.library],
            filename=filename,
            code=rendered.code,
            annotations=self._annotations(rendered),
            uses=[
                ConceptCodeUseOut(
                    concept=self._concept_refs([use.concept])[0],
                    filename=f"{code_module(use.concept, code.library)}.py",
                    names=list(use.names),
                )
                for use in code.uses
            ],
        )

    def _code(self, variant: str, title: str, rendered: RenderedCode, source: str) -> CodeOut:
        return CodeOut.model_validate(
            {
                "variant": variant,
                "title": title,
                "filename": rendered.filename,
                "code": rendered.code,
                "source": source,
                "annotations": self._annotations(rendered),
            }
        )

    def _annotations(self, rendered: RenderedCode) -> list[CodeAnnotationOut]:
        return [
            CodeAnnotationOut(
                anchor=s.anchor,
                start_line=s.start_line,
                end_line=s.end_line,
                short=s.annotation.short,
                deep=s.annotation.deep,
                concepts=self._concept_refs(s.annotation.concepts),
            )
            for s in rendered.spans
        ]

    def _family(self, family: Family) -> FamilyOut:
        entries = []
        for entry in family.meta.algorithms:
            pack = self._registry.get(entry.id) if entry.id in self._registry else None
            entries.append(
                FamilyAlgorithmOut(
                    id=entry.id,
                    name=entry.name,
                    available=pack is not None,
                    summary=self._summary(entry.id) if pack else None,
                    task_types=sorted(pack.spec.task_types) if pack else [],
                    difficulty=pack.spec.difficulty if pack else None,
                )
            )
        return FamilyOut(
            id=family.id, title=family.meta.title, summary=family.meta.summary, algorithms=entries
        )

    def _summary(self, algorithm_id: str) -> str | None:
        try:
            return self._catalog.algorithm(algorithm_id).meta.summary
        except UnknownContentError:
            return None

    def _algorithms_using(self, concept_id: str) -> list[AlgorithmRefOut]:
        return [
            AlgorithmRefOut(id=p.spec.id, name=p.spec.name)
            for p in self._registry
            if concept_id in p.spec.concepts
        ]

    def _concept_refs(self, ids: Sequence[str]) -> list[ConceptRefOut]:
        return [ConceptRefOut(id=i, title=self._concept(i).meta.title) for i in ids]

    def _pack(self, algorithm_id: str) -> AlgorithmPack:
        try:
            return self._registry.get(algorithm_id)
        except UnknownAlgorithmError:
            raise NotFoundError(f"Unknown algorithm {algorithm_id!r}.") from None

    def _concept(self, concept_id: str) -> Concept:
        try:
            return self._catalog.concept(concept_id)
        except UnknownContentError:
            raise NotFoundError(f"Unknown concept {concept_id!r}.") from None


def _resources(links: Sequence[ResourceLink]) -> list[ResourceOut]:
    return [
        ResourceOut(
            type=link.resource.type,
            title=link.resource.title,
            url=link.resource.url,
            author=link.resource.author,
            level=link.resource.level,
            duration_min=link.resource.duration_min,
            why=link.resource.why,
            note=link.note,
        )
        for link in links
    ]
