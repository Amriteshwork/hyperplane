from pathlib import Path

import pytest

from app.content.catalog import ContentCatalog, ContentError, UnknownContentError
from app.content.gaps import content_gaps, missing_sections

CONCEPT = "---\ntitle: MSE\nsummary: Squared error.\nprerequisites: [loss]\n---\n\n# Body\n"
RESOURCE = """
type: video
title: MSE explained
url: https://example.org/v
level: beginner
why: Short and clear.
source: curated
checked: 2026-10-01
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_mse(root: Path, resource_ids: str = "[mse-video]") -> None:
    write(root / "concepts" / "mse" / "concept.md", CONCEPT)
    write(root / "concepts" / "mse" / "resources.yaml", resource_ids)
    write(root / "resources" / "mse-video.yaml", RESOURCE)


def test_loads_concepts_with_frontmatter_and_shared_resources(tmp_path: Path) -> None:
    write_mse(tmp_path)

    catalog = ContentCatalog.load(tmp_path)
    concept = catalog.concept("mse")

    assert concept.meta.title == "MSE"
    assert concept.meta.prerequisites == ("loss",)
    assert concept.body == "# Body\n"
    assert concept.resources[0].id == "mse-video"
    assert concept.resources[0].title == "MSE explained"
    assert catalog.resources() == [concept.resources[0]]


@pytest.mark.parametrize(
    ("file", "text", "match"),
    [
        ("concepts/mse/concept.md", "no frontmatter", "frontmatter"),
        ("concepts/mse/concept.md", "---\ntitle: X\n---\n", "invalid frontmatter"),
        ("concepts/mse/resources.yaml", "- type: podcast\n", "invalid resources"),
        ("concepts/mse/resources.yaml", "[missing]", "unknown resources"),
        ("concepts/mse/resources.yaml", "[mse-video, mse-video]", "duplicate"),
        ("resources/mse-video.yaml", "type: podcast\n", "invalid resource"),
        ("resources/mse-video.yaml", "- a list\n", "mapping"),
    ],
)
def test_malformed_files_fail_loudly(tmp_path: Path, file: str, text: str, match: str) -> None:
    write_mse(tmp_path)
    write(tmp_path / file, text)
    with pytest.raises(ContentError, match=match):
        ContentCatalog.load(tmp_path)


def test_missing_root_is_empty_and_unknown_ids_raise(tmp_path: Path) -> None:
    catalog = ContentCatalog.load(tmp_path / "nope")
    assert catalog.families() == []
    with pytest.raises(UnknownContentError):
        catalog.concept("mse")


def test_gaps_and_sections(tmp_path: Path) -> None:
    write_mse(tmp_path)

    gaps = content_gaps(ContentCatalog.load(tmp_path), [])

    assert [g.message for g in gaps] == ["needs a beginner blog/post"]  # the video is there
    assert "Math" in missing_sections("## Intuition\n")
