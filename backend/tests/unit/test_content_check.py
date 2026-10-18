from pathlib import Path

import pytest

from scripts.content_check import Level, check_layout, main


@pytest.fixture
def content(tmp_path: Path) -> Path:
    root = tmp_path / "content"
    for name in ("overview.md", "resources.yaml"):
        _touch(root / "algorithms" / "linear_regression" / name)
    for name in ("concept.md", "resources.yaml"):
        _touch(root / "concepts" / "mse" / name)
    _touch(root / "concepts" / "mse" / "posts" / "intro.md")
    _touch(root / "families" / "linear-models.md")
    _touch(root / "README.md")
    return root


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_valid_layout_has_no_issues(content: Path) -> None:
    assert check_layout(content) == []
    assert main(["--content-dir", str(content)]) == 0


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
    assert main(["--content-dir", str(content)]) == 1


def test_unexpected_entries(content: Path) -> None:
    _touch(content / "stray.txt")
    _touch(content / "algorithms" / "loose-file.md")
    _touch(content / "families" / "notes.txt")

    levels = {i.path.name: i.level for i in check_layout(content)}

    assert levels == {
        "stray.txt": Level.WARNING,
        "loose-file.md": Level.ERROR,
        "notes.txt": Level.ERROR,
    }
