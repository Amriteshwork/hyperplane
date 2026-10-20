"""Schemas of the files in `content/` (see content/README.md)."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

_SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"
_ALGO_ID = r"^[a-z][a-z0-9_]*$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ResourceType(StrEnum):
    BLOG = "blog"
    VIDEO = "video"
    PAPER = "paper"
    INTERACTIVE = "interactive"
    COURSE = "course"


class Level(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Resource(_Frozen):
    """One `resources/<id>.yaml` file; algorithms and concepts reference it by id."""

    id: str = Field(pattern=_SLUG)
    type: ResourceType
    title: str
    url: str = Field(pattern=r"^https://")
    author: str | None = None
    level: Level
    duration_min: int | None = Field(default=None, gt=0)
    why: str
    source: str = Field(pattern=r"^(curated|own)$")
    checked: date


class FamilyAlgorithm(_Frozen):
    """An algorithm the family will contain, whether or not its pack exists yet."""

    id: str = Field(pattern=_ALGO_ID)
    name: str


class FamilyFrontmatter(_Frozen):
    title: str
    summary: str
    order: int = Field(ge=1)
    algorithms: tuple[FamilyAlgorithm, ...]


class AlgorithmFrontmatter(_Frozen):
    title: str
    summary: str
    difficulty: int = Field(ge=1, le=5)


class ConceptFrontmatter(_Frozen):
    title: str
    summary: str
    prerequisites: tuple[str, ...] = ()


class Family(_Frozen):
    id: str = Field(pattern=_SLUG)
    meta: FamilyFrontmatter
    body: str


class AlgorithmContent(_Frozen):
    id: str = Field(pattern=_ALGO_ID)
    meta: AlgorithmFrontmatter
    body: str
    resources: tuple[Resource, ...]


class Concept(_Frozen):
    id: str = Field(pattern=_SLUG)
    meta: ConceptFrontmatter
    body: str
    resources: tuple[Resource, ...]
