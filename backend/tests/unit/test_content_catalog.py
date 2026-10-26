from collections.abc import Callable
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


def test_curriculum_lists_algorithms_family_by_family_in_page_order(tmp_path: Path) -> None:
    family = "---\ntitle: {t}\nsummary: s\norder: {o}\nalgorithms:\n{a}\n---\n"
    entry = "  - {{ id: {i}, name: N }}"
    write(
        tmp_path / "families" / "a.md",
        family.format(t="A", o=2, a=entry.format(i="trees")),
    )
    write(
        tmp_path / "families" / "b.md",
        family.format(t="B", o=1, a="\n".join(entry.format(i=i) for i in ("ols", "lasso"))),
    )

    assert ContentCatalog.load(tmp_path).curriculum() == ["ols", "lasso", "trees"]


def test_gaps_and_sections(tmp_path: Path) -> None:
    write_mse(tmp_path)

    gaps = content_gaps(ContentCatalog.load(tmp_path), [])

    assert [g.message for g in gaps if g.item == "concepts/mse"] == [  # the video is there
        "needs a beginner blog/post",
        "needs a NumPy implementation",
        "needs a PyTorch implementation",
    ]
    assert "Math" in missing_sections("## Intuition\n")


TRACK = """---
title: Squared error, from scratch
summary: Everything MSE needs.
order: 1
goal: mse
chapters:
  - {title: Basics, summary: What a loss is., concepts: [loss]}
  - {title: The goal, summary: Squares., concepts: [%s]}
---

Intro.
"""


def test_tracks_load_and_must_be_a_complete_route_to_their_goal(tmp_path: Path) -> None:
    write_mse(tmp_path)
    write(tmp_path / "tracks" / "mse" / "track.md", TRACK % "mse")
    write(tmp_path / "tracks" / "mse" / "resources.yaml", LINKS)

    [track] = ContentCatalog.load(tmp_path).tracks()
    assert [c.title for c in track.meta.chapters] == ["Basics", "The goal"]
    assert track.resources[0].resource.id == "mse-video"

    write(
        tmp_path / "families" / "losses.md",
        "---\ntitle: L\nsummary: s\norder: 1\ntracks: [nope]\n---\n",
    )
    with pytest.raises(ContentError, match="unknown tracks \\['nope'\\]"):
        ContentCatalog.load(tmp_path)

    write(
        tmp_path / "families" / "losses.md",
        "---\ntitle: L\nsummary: s\norder: 1\ntracks: [mse]\n---\n",
    )
    [family] = ContentCatalog.load(tmp_path).families()
    assert family.meta.tracks == ("mse",)
    assert family.meta.algorithms == ()

    write(tmp_path / "tracks" / "mse" / "track.md", TRACK % "")
    with pytest.raises(ContentError, match="tracks/mse"):
        ContentCatalog.load(tmp_path)


PAPER = """---
title: The MSE paper, explained
summary: Why squares.
order: 1
source: %s
concepts: [mse]
videos:
  squared-area: {title: Squares as areas, youtube: dQw4w9WgXcQ}
---

Why read it.

## 1 · Introduction

The idea.

::video[squared-area]

## 2 · Results

It worked.
"""
SCENE = "from manim import Scene\n\n\nclass SquaredArea(Scene):\n    pass\n"


def write_paper(root: Path, source: str = "mse-paper") -> Path:
    write_mse(root)
    write(root / "resources" / "mse-paper.yaml", RESOURCE.replace("type: video", "type: paper"))
    folder = root / "papers" / "mse"
    write(folder / "paper.md", PAPER % source)
    write(folder / "resources.yaml", LINKS)
    write(folder / "scenes" / "squared_area.py", SCENE)
    return folder


def test_papers_split_into_sections_with_videos_backed_by_scenes(tmp_path: Path) -> None:
    write_paper(tmp_path)

    [paper] = ContentCatalog.load(tmp_path).papers()

    assert paper.source.id == "mse-paper"
    assert paper.intro == "Why read it.\n"
    assert [(s.id, s.title) for s in paper.sections] == [
        ("1-introduction", "1 · Introduction"),
        ("2-results", "2 · Results"),
    ]
    assert "::video[squared-area]" in paper.sections[0].body
    assert paper.meta.videos["squared-area"].youtube == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda f: (f / "scenes" / "squared_area.py").unlink(), "not with a scene"),
        (lambda f: write(f / "scenes" / "extra.py", ""), "with a scene but not declared"),
        (lambda f: write(f / "scenes" / "squared_area.py", "class Other: ..."), "SquaredArea"),
        (lambda f: _edit(f, "::video[squared-area]", ""), "not used in the text"),
        (
            lambda f: _edit(f, "It worked.", "It worked.\n\n::video[nope]"),
            "used in the text but not declared",
        ),
        (lambda f: _edit(f, "2 · Results", "1 · Introduction"), "same heading"),
        (lambda f: _edit(f, "[mse]", "[nope]"), "unknown concepts"),
        (lambda f: _edit(f, "videos:", "track: nope\nvideos:"), "unknown track"),
        (lambda f: _edit(f, "dQw4w9WgXcQ", "not-an-id"), "invalid frontmatter"),
    ],
)
def test_invalid_papers_fail_loading(
    tmp_path: Path, change: Callable[[Path], object], match: str
) -> None:
    change(write_paper(tmp_path))
    with pytest.raises(ContentError, match=match):
        ContentCatalog.load(tmp_path)


@pytest.mark.parametrize("source", ["mse-video", "missing"])
def test_paper_source_must_be_a_paper_or_blog(tmp_path: Path, source: str) -> None:
    write_paper(tmp_path, source)
    with pytest.raises(ContentError, match="not a paper or blog"):
        ContentCatalog.load(tmp_path)


def _edit(folder: Path, old: str, new: str) -> None:
    path = folder / "paper.md"
    path.write_text(path.read_text().replace(old, new))
