"""Learning-content API contract: families, algorithms, concepts, explained code."""

from typing import Literal

from pydantic import BaseModel, JsonValue

from app.content.models import Level, ResourceType
from app.ml.types import TaskType


class ResourceOut(BaseModel):
    type: ResourceType
    title: str
    url: str
    author: str | None
    level: Level
    duration_min: int | None
    why: str


class ConceptRefOut(BaseModel):
    id: str
    title: str


class AlgorithmRefOut(BaseModel):
    id: str
    name: str


class FamilyAlgorithmOut(BaseModel):
    id: str
    name: str
    #: A pack is registered; upcoming algorithms are listed but can't be opened yet.
    available: bool
    summary: str | None
    task_types: list[TaskType]
    difficulty: int | None


class FamilyOut(BaseModel):
    id: str
    title: str
    summary: str
    algorithms: list[FamilyAlgorithmOut]


class CodeVariantOut(BaseModel):
    id: str
    title: str
    filename: str


class AlgorithmOut(BaseModel):
    id: str
    name: str
    summary: str
    family: FamilyOut | None
    task_types: list[TaskType]
    difficulty: int
    #: `overview.md` body (Markdown + KaTeX).
    overview: str
    concepts: list[ConceptRefOut]
    resources: list[ResourceOut]
    #: JSON Schema of the hyperparameters (Pydantic), rendered by the frontend form.
    params_schema: dict[str, JsonValue]
    code_variants: list[CodeVariantOut]


class CodeAnnotationOut(BaseModel):
    anchor: str
    #: 1-based, inclusive lines of `code`.
    start_line: int
    end_line: int
    short: str
    deep: str
    concepts: list[ConceptRefOut]


class CodeOut(BaseModel):
    variant: str
    title: str
    filename: str
    code: str
    annotations: list[CodeAnnotationOut]
    #: `example`: rendered with placeholder settings; `run`: the exact code of a run.
    source: Literal["example", "run"]


class ConceptOut(BaseModel):
    id: str
    title: str
    summary: str
    body: str
    prerequisites: list[ConceptRefOut]
    resources: list[ResourceOut]
    algorithms: list[AlgorithmRefOut]
