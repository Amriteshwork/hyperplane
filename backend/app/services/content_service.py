"""Content queries: the learning catalog joined with the registered packs."""

from collections.abc import Sequence

from app.content.catalog import ContentCatalog, UnknownContentError
from app.content.models import Concept, Family, Resource
from app.core.errors import NotFoundError
from app.ml.base import AlgorithmPack
from app.ml.codegen import CodegenContext, RenderedCode, render
from app.ml.registry import AlgorithmRegistry, UnknownAlgorithmError
from app.schemas.content import (
    AlgorithmOut,
    AlgorithmRefOut,
    CodeAnnotationOut,
    CodeOut,
    CodeVariantOut,
    ConceptOut,
    ConceptRefOut,
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
            {**_EXAMPLE, "params": pack.params_model().model_dump(mode="json")}
        )
        return self._code(variant, template.title, render(template, context), "example")

    def get_concept(self, concept_id: str) -> ConceptOut:
        concept = self._concept(concept_id)
        return ConceptOut(
            id=concept.id,
            title=concept.meta.title,
            summary=concept.meta.summary,
            body=concept.body,
            prerequisites=self._concept_refs(concept.meta.prerequisites),
            resources=_resources(concept.resources),
            algorithms=[
                AlgorithmRefOut(id=p.spec.id, name=p.spec.name)
                for p in self._registry
                if concept_id in p.spec.concepts
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
                "annotations": [
                    CodeAnnotationOut(
                        anchor=s.anchor,
                        start_line=s.start_line,
                        end_line=s.end_line,
                        short=s.annotation.short,
                        deep=s.annotation.deep,
                        concepts=self._concept_refs(s.annotation.concepts),
                    )
                    for s in rendered.spans
                ],
            }
        )

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


def _resources(resources: Sequence[Resource]) -> list[ResourceOut]:
    return [
        ResourceOut(
            type=r.type,
            title=r.title,
            url=r.url,
            author=r.author,
            level=r.level,
            duration_min=r.duration_min,
            why=r.why,
        )
        for r in resources
    ]
