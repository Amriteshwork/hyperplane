from pathlib import Path

import pytest

from app.content.catalog import ContentCatalog, ContentError, UnknownContentError
from app.content.gaps import content_gaps, missing_sections
from app.content.models import CodeLibrary

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


def test_concept_code_loads_in_library_order(tmp_path: Path) -> None:
    write_mse(tmp_path)
    write(tmp_path / "concepts" / "mse" / "code" / "pytorch.py", "import torch\n")
    write(tmp_path / "concepts" / "mse" / "code" / "numpy.py", "import numpy\n")

    code = ContentCatalog.load(tmp_path).concept("mse").code

    assert [(c.library, c.source) for c in code] == [
        (CodeLibrary.NUMPY, "import numpy\n"),
        (CodeLibrary.PYTORCH, "import torch\n"),
    ]


ANNOTATIONS = """
numpy:
  fit:
    short: Fit it.
    deep: Fit it, in depth.
"""


def test_concept_code_annotations_match_their_anchors(tmp_path: Path) -> None:
    write_mse(tmp_path)
    code = tmp_path / "concepts" / "mse" / "code"
    write(code / "numpy.py", "import numpy\n# @a:fit\nbeta = 1\n")
    write(code / "annotations.yaml", ANNOTATIONS)

    (numpy,) = ContentCatalog.load(tmp_path).concept("mse").code

    assert numpy.annotations["fit"].short == "Fit it."


@pytest.mark.parametrize(
    ("source", "annotations", "match"),
    [
        ("# @a:fit\nbeta = 1\n# @a:extra\nb = 2\n", ANNOTATIONS, "without annotations"),
        ("beta = 1\n", ANNOTATIONS, "without anchors"),
        ("# @a:fit\nbeta = 1\n", ANNOTATIONS + "pytorch: {}\n", "sections without code"),
        ("beta = 1\n", "jax: {}\n", "invalid annotations"),
    ],
)
def test_concept_code_annotation_mismatches_fail(
    tmp_path: Path, source: str, annotations: str, match: str
) -> None:
    write_mse(tmp_path)
    write(tmp_path / "concepts" / "mse" / "code" / "numpy.py", source)
    write(tmp_path / "concepts" / "mse" / "code" / "annotations.yaml", annotations)
    with pytest.raises(ContentError, match=match):
        ContentCatalog.load(tmp_path)


def test_concept_code_must_be_a_known_library(tmp_path: Path) -> None:
    write_mse(tmp_path)
    write(tmp_path / "concepts" / "mse" / "code" / "jax.py", "import jax\n")
    with pytest.raises(ContentError, match="expected one of"):
        ContentCatalog.load(tmp_path)


def test_missing_root_is_empty_and_unknown_ids_raise(tmp_path: Path) -> None:
    catalog = ContentCatalog.load(tmp_path / "nope")
    assert catalog.families() == []
    with pytest.raises(UnknownContentError):
        catalog.concept("mse")


def test_gaps_and_sections(tmp_path: Path) -> None:
    write_mse(tmp_path)

    gaps = content_gaps(ContentCatalog.load(tmp_path), [])

    assert [g.message for g in gaps] == [  # the video is there
        "needs a beginner blog/post",
        "needs a NumPy implementation",
        "needs a PyTorch implementation",
    ]
    assert "Math" in missing_sections("## Intuition\n")
