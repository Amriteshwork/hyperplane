"""Schemas of the files in `content/` (see content/README.md)."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.ml.base import Annotation

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


class ResourceLink(_Frozen):
    """A resource as listed on one page (`resources.yaml`), with what it teaches there (ADR 0016)."""

    resource: Resource
    note: str = Field(min_length=1)


class FamilyAlgorithm(_Frozen):
    """An algorithm the family will contain, whether or not its pack exists yet."""

    id: str = Field(pattern=_ALGO_ID)
    name: str


class FamilyFrontmatter(_Frozen):
    title: str
    summary: str
    order: int = Field(ge=1)
    algorithms: tuple[FamilyAlgorithm, ...] = ()
    #: Guided tracks shown in this family on Learn, for subjects taught as concepts, not packs.
    tracks: tuple[str, ...] = ()


class AlgorithmFrontmatter(_Frozen):
    title: str
    summary: str
    difficulty: int = Field(ge=1, le=5)


class CodeLibrary(StrEnum):
    """Libraries a concept is implemented in from scratch: `code/<library>.py`, in display order."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"


CODE_LIBRARY_TITLES: dict[CodeLibrary, str] = {
    CodeLibrary.NUMPY: "NumPy",
    CodeLibrary.PYTORCH: "PyTorch",
}

#: What to `pip install` to run a library's concept code.
CODE_LIBRARY_PACKAGES: dict[CodeLibrary, str] = {
    CodeLibrary.NUMPY: "numpy",
    CodeLibrary.PYTORCH: "torch",
}


class ConceptArea(StrEnum):
    """Where a concept sits on the concept index, in display order."""

    FOUNDATIONS = "foundations"
    OPTIMISATION = "optimisation"
    EVALUATION = "evaluation"
    FEATURES = "features"
    NEURAL_NETWORKS = "neural-networks"
    SEQUENCE_MODELS = "sequence-models"


CONCEPT_AREA_TITLES: dict[ConceptArea, str] = {
    ConceptArea.FOUNDATIONS: "Maths & NumPy foundations",
    ConceptArea.OPTIMISATION: "Loss & optimisation",
    ConceptArea.EVALUATION: "Evaluation & generalisation",
    ConceptArea.FEATURES: "Features & regularisation",
    ConceptArea.NEURAL_NETWORKS: "Neural networks",
    ConceptArea.SEQUENCE_MODELS: "Sequence models & transformers",
}


def code_module(concept_id: str, library: CodeLibrary) -> str:
    """The name a concept's code is shown, downloaded and imported as (`normal_equation_numpy`).

    Never plain `numpy`: run from its folder, a file named `numpy.py` would shadow NumPy itself.
    """
    return f"{concept_id.replace('-', '_')}_{library.value}"


class CodeUse(_Frozen):
    """Names one concept's code imports from a prerequisite's code (ADR 0016)."""

    concept: str
    names: tuple[str, ...]


class ConceptCode(_Frozen):
    """A short, runnable implementation of a concept on random data.

    `source` keeps its `# @a:<anchor>` comments; `annotations` explain them, exactly like a
    pack's templates (ADR 0006).
    """

    library: CodeLibrary
    source: str
    annotations: dict[str, Annotation] = {}
    uses: tuple[CodeUse, ...] = ()


class ConceptFrontmatter(_Frozen):
    title: str
    summary: str
    area: ConceptArea
    prerequisites: tuple[str, ...] = ()


class TrackChapter(_Frozen):
    title: str
    summary: str
    concepts: tuple[str, ...] = Field(min_length=1)


class TrackFrontmatter(_Frozen):
    title: str
    summary: str
    order: int = Field(ge=1)
    #: The concept the track leads to; its chapters cover it and everything it builds on.
    goal: str
    chapters: tuple[TrackChapter, ...] = Field(min_length=1)


class Family(_Frozen):
    id: str = Field(pattern=_SLUG)
    meta: FamilyFrontmatter
    body: str


class AlgorithmContent(_Frozen):
    id: str = Field(pattern=_ALGO_ID)
    meta: AlgorithmFrontmatter
    body: str
    resources: tuple[ResourceLink, ...]


class Concept(_Frozen):
    id: str = Field(pattern=_SLUG)
    meta: ConceptFrontmatter
    body: str
    resources: tuple[ResourceLink, ...]
    code: tuple[ConceptCode, ...] = ()


class Track(_Frozen):
    """A guided route to one concept: its whole learning path, cut into authored chapters."""

    id: str = Field(pattern=_SLUG)
    meta: TrackFrontmatter
    body: str
    resources: tuple[ResourceLink, ...]


class PaperVideo(_Frozen):
    """A Manim animation (`scenes/<id>.py`), rendered offline and hosted on YouTube (ADR 0022)."""

    title: str
    #: Filled in once the render is uploaded; until then the page says the clip is coming.
    youtube: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{11}$")


class PaperFrontmatter(_Frozen):
    title: str
    summary: str
    order: int = Field(ge=1)
    #: The paper or blog post walked through: a resource id of type paper or blog.
    source: str
    #: Concepts the walkthrough leans on; the page lists them under "Before you read".
    concepts: tuple[str, ...] = Field(min_length=1)
    #: A guided track that teaches those concepts from scratch.
    track: str | None = None
    videos: dict[str, PaperVideo] = {}


class PaperSection(_Frozen):
    """One `## ` section of a walkthrough, usually one section of the original."""

    id: str = Field(pattern=_SLUG)
    title: str
    body: str


class Paper(_Frozen):
    """A section-by-section, plain-words walkthrough of one paper or blog post."""

    id: str = Field(pattern=_SLUG)
    meta: PaperFrontmatter
    source: Resource
    #: Text before the first section.
    intro: str
    sections: tuple[PaperSection, ...] = Field(min_length=1)
    resources: tuple[ResourceLink, ...]
