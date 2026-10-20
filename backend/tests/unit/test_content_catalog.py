from pathlib import Path

import pytest

from app.content.catalog import ContentCatalog, ContentError, UnknownContentError
from app.content.gaps import content_gaps, missing_sections
from app.content.models import CodeLibrary

CONCEPT = "---\ntitle: MSE\nsummary: Squared error.\narea: optimisation\nprerequisites: [loss]\n---\n\n# Body\n"
LOSS = "---\ntitle: Loss\nsummary: How wrong.\narea: optimisation\n---\n"
LINKS = "[{id: mse-video, note: The squared-error picture.}]"
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


def write_mse(root: Path, links: str = LINKS) -> None:
    write(root / "concepts" / "loss" / "concept.md", LOSS)
    write(root / "concepts" / "loss" / "resources.yaml", "[]")
    write(root / "concepts" / "mse" / "concept.md", CONCEPT)
    write(root / "concepts" / "mse" / "resources.yaml", links)
    write(root / "resources" / "mse-video.yaml", RESOURCE)


def test_loads_concepts_with_frontmatter_and_shared_resources(tmp_path: Path) -> None:
    write_mse(tmp_path)

    catalog = ContentCatalog.load(tmp_path)
    concept = catalog.concept("mse")

    assert concept.meta.title == "MSE"
    assert concept.meta.prerequisites == ("loss",)
    assert concept.body == "# Body\n"
    [link] = concept.resources
    assert link.resource.id == "mse-video"
    assert link.resource.title == "MSE explained"
    assert link.note == "The squared-error picture."
    assert catalog.resources() == [link.resource]


@pytest.mark.parametrize(
    ("file", "text", "match"),
    [
        ("concepts/mse/concept.md", "no frontmatter", "frontmatter"),
        ("concepts/mse/concept.md", "---\ntitle: X\n---\n", "invalid frontmatter"),
        (
            "concepts/mse/concept.md",
            "---\ntitle: M\nsummary: s\narea: astrology\n---\n",
            "invalid frontmatter",
        ),
        ("concepts/mse/resources.yaml", "- type: podcast\n", "invalid resources"),
        ("concepts/mse/resources.yaml", "[mse-video]", "invalid resources"),  # no note
        ("concepts/mse/resources.yaml", "[{id: mse-video, note: ''}]", "invalid resources"),
        ("concepts/mse/resources.yaml", "[{id: missing, note: n}]", "unknown resources"),
        ("concepts/mse/resources.yaml", f"[{LINKS[1:-1]}, {LINKS[1:-1]}]", "duplicate"),
        (
            "concepts/mse/concept.md",
            "---\ntitle: M\nsummary: s\narea: evaluation\nprerequisites: [x]\n---\n",
            "unknown",
        ),
        (
            "concepts/loss/concept.md",
            "---\ntitle: L\nsummary: s\narea: evaluation\nprerequisites: [mse]\n---\n",
            "cycle",
        ),
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


def write_code_chain(root: Path) -> None:
    """vec ← mat ← solve, and solve also imports vec directly."""
    for concept, prerequisites, source in [
        ("vec", "[]", "def dot(u, v):\n    return 0\n"),
        (
            "mat",
            "[vec]",
            "from vec_numpy import dot\n\n\ndef matmul(a, b):\n    return dot(a, b)\n",
        ),
        (
            "solve",
            "[mat]",
            "from mat_numpy import matmul\nfrom vec_numpy import dot\n\nx = matmul(1, dot(2, 3))\n",
        ),
    ]:
        meta = f"---\ntitle: {concept}\nsummary: s\narea: foundations\nprerequisites: {prerequisites}\n---\n"
        write(root / "concepts" / concept / "concept.md", meta)
        write(root / "concepts" / concept / "resources.yaml", "[]")
        write(root / "concepts" / concept / "code" / "numpy.py", source)


def test_code_with_dependencies_follows_imports_transitively_once_each(tmp_path: Path) -> None:
    write_code_chain(tmp_path)
    catalog = ContentCatalog.load(tmp_path)

    files = catalog.code_with_dependencies("solve", CodeLibrary.NUMPY)

    assert [concept for concept, _ in files] == ["solve", "mat", "vec"]
    assert [c for c, _ in catalog.code_with_dependencies("vec", CodeLibrary.NUMPY)] == ["vec"]
    with pytest.raises(UnknownContentError, match="no pytorch code"):
        catalog.code_with_dependencies("solve", CodeLibrary.PYTORCH)


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

    assert [g.message for g in gaps if g.item == "concepts/mse"] == [  # the video is there
        "needs a beginner blog/post",
        "needs a NumPy implementation",
        "needs a PyTorch implementation",
    ]
    assert "Math" in missing_sections("## Intuition\n")
