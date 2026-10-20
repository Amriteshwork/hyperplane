"""What `content_check` reports beyond schema errors: missing sections and curation gaps."""

import re
from dataclasses import dataclass

from app.content.catalog import ContentCatalog
from app.content.models import CODE_LIBRARY_TITLES, Level, ResourceType

OVERVIEW_SECTIONS = (
    "Intuition",
    "How it works",
    "Math",
    "Assumptions",
    "Hyperparameters",
    "When to use / not use",
    "Common pitfalls",
    "Interview questions",
)
_HEADING = re.compile(r"^## (.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Gap:
    """Something to write, record or curate; becomes a content-backlog entry."""

    item: str
    message: str


def missing_sections(body: str) -> list[str]:
    headings = set(_HEADING.findall(body))
    return [s for s in OVERVIEW_SECTIONS if s not in headings]


def content_gaps(catalog: ContentCatalog, algorithm_ids: list[str]) -> list[Gap]:
    """Gap policy (ARCHITECTURE §8): each concept needs a beginner read, a video and code."""
    gaps: list[Gap] = []
    for concept in catalog.concepts():
        reads = [
            r
            for r in concept.resources
            if r.type is ResourceType.BLOG and r.level is Level.BEGINNER
        ]
        videos = [r for r in concept.resources if r.type is ResourceType.VIDEO]
        if not reads:
            gaps.append(Gap(f"concepts/{concept.id}", "needs a beginner blog/post"))
        if not videos:
            gaps.append(Gap(f"concepts/{concept.id}", "needs a video"))
        written = {c.library for c in concept.code}
        for library, title in CODE_LIBRARY_TITLES.items():
            if library not in written:
                gaps.append(Gap(f"concepts/{concept.id}", f"needs a {title} implementation"))
    for algorithm_id in algorithm_ids:
        if not catalog.algorithm(algorithm_id).resources:
            gaps.append(Gap(f"algorithms/{algorithm_id}", "has no curated resources"))
    return gaps
