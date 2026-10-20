from pathlib import Path

import pytest

from app.content.gaps import OVERVIEW_SECTIONS
from app.ml.registry import AlgorithmRegistry
from scripts.content_check import Level, check_layout, main

NO_PACKS = AlgorithmRegistry()
OVERVIEW = "---\ntitle: LR\nsummary: s\ndifficulty: 1\n---\n" + "".join(
    f"## {section}\n\ntext\n\n" for section in OVERVIEW_SECTIONS
)
CONCEPT = "---\ntitle: MSE\nsummary: s\narea: optimisation\n---\nBody.\n"
FAMILY = (
    "---\ntitle: Linear\nsummary: s\norder: 1\n"
    "algorithms: [{id: linear_regression, name: LR}]\n---\nBody.\n"
)


@pytest.fixture
def content(tmp_path: Path) -> Path:
    root = tmp_path / "content"
    _touch(root / "algorithms" / "linear_regression" / "overview.md", OVERVIEW)
    _touch(root / "algorithms" / "linear_regression" / "resources.yaml", "[]")
    _touch(root / "concepts" / "mse" / "concept.md", CONCEPT)
    _touch(root / "concepts" / "mse" / "resources.yaml", "[]")
    _touch(root / "concepts" / "mse" / "posts" / "intro.md")
    _touch(root / "families" / "linear-models.md", FAMILY)
    _touch(root / "resources" / ".gitkeep")
    _touch(root / "README.md")
    return root


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_valid_layout_has_no_issues(content: Path) -> None:
    assert check_layout(content) == []
    assert main(["--content-dir", str(content)], NO_PACKS) == 0


def test_invalid_frontmatter_is_an_error(content: Path) -> None:
    _touch(content / "concepts" / "mse" / "concept.md", "no frontmatter")
    assert main(["--content-dir", str(content)], NO_PACKS) == 1


def test_real_content_passes_with_backlog(tmp_path: Path) -> None:
    backlog = tmp_path / "backlog.md"
    assert main(["--backlog", str(backlog)]) == 0
    assert "- [ ] `concepts/" in backlog.read_text()


def test_missing_root(tmp_path: Path) -> None:
    [issue] = check_layout(tmp_path / "nope")
    assert issue.level is Level.ERROR


def test_missing_section_and_files(content: Path) -> None:
    (content / "families" / "linear-models.md").unlink()
    (content / "families").rmdir()
    (content / "concepts" / "mse" / "concept.md").unlink()

    problems = {(i.level, i.path.relative_to(content).as_posix()) for i in check_layout(content)}

    assert problems == {
        (Level.ERROR, "families"),
        (Level.ERROR, "concepts/mse/concept.md"),
    }
    assert main(["--content-dir", str(content)], NO_PACKS) == 1


def test_unexpected_entries(content: Path) -> None:
    _touch(content / "stray.txt")
    _touch(content / "algorithms" / "loose-file.md")
    _touch(content / "families" / "notes.txt")
    _touch(content / "resources" / "notes.md")

    levels = {i.path.name: i.level for i in check_layout(content)}

    assert levels == {
        "stray.txt": Level.WARNING,
        "loose-file.md": Level.ERROR,
        "notes.txt": Level.ERROR,
        "notes.md": Level.ERROR,
    }


def test_unreferenced_resource_is_a_warning(
    content: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(
        content / "resources" / "orphan.yaml",
        "type: blog\ntitle: T\nurl: https://example.org\nlevel: beginner\n"
        "why: w\nsource: curated\nchecked: 2026-10-20\n",
    )

    assert main(["--content-dir", str(content)], NO_PACKS) == 0
    assert "resources/orphan.yaml: not referenced" in capsys.readouterr().out


def test_unknown_resource_id_is_an_error(content: Path) -> None:
    _touch(content / "concepts" / "mse" / "resources.yaml", "[nope]")
    assert main(["--content-dir", str(content)], NO_PACKS) == 1
