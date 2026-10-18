"""Validate the content/ folder layout (ARCHITECTURE §8).

Stub: checks folders and required files only. Frontmatter, resources.yaml schema and the
content-gap backlog arrive with the M0 "Content system" milestone.

Usage: uv run python -m scripts.content_check [--content-dir PATH]
"""

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.core.config import get_settings

SECTIONS = ("algorithms", "concepts", "families")
REQUIRED_FILES = {
    "algorithms": ("overview.md", "resources.yaml"),
    "concepts": ("concept.md", "resources.yaml"),
}
IGNORED = {"README.md", ".gitkeep", ".DS_Store"}


class Level(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Issue:
    level: Level
    path: Path
    message: str


def check_layout(root: Path) -> list[Issue]:
    """Return every layout problem under `root`, errors and warnings alike."""
    if not root.is_dir():
        return [Issue(Level.ERROR, root, "content directory not found")]

    issues: list[Issue] = []
    for entry in sorted(root.iterdir()):
        if entry.name not in IGNORED and entry.name not in SECTIONS:
            issues.append(Issue(Level.WARNING, entry, "unexpected entry"))

    for section in SECTIONS:
        if not (root / section).is_dir():
            issues.append(Issue(Level.ERROR, root / section, "missing section folder"))

    for section, required in REQUIRED_FILES.items():
        folder = root / section
        if not folder.is_dir():
            continue
        for item in sorted(p for p in folder.iterdir() if p.name not in IGNORED):
            if not item.is_dir():
                issues.append(Issue(Level.ERROR, item, f"expected a folder per {section[:-1]}"))
                continue
            for name in required:
                if not (item / name).is_file():
                    issues.append(Issue(Level.ERROR, item / name, "missing required file"))

    families = root / "families"
    if families.is_dir():
        for item in sorted(p for p in families.iterdir() if p.name not in IGNORED):
            if not (item.is_file() and item.suffix == ".md"):
                issues.append(Issue(Level.ERROR, item, "families/ holds <family_id>.md files"))

    return issues


def main(argv: list[str] | None = None) -> int:
    """Print issues; exit 1 if any is an error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--content-dir", type=Path, default=get_settings().content_dir)
    args = parser.parse_args(argv)

    root: Path = args.content_dir.resolve()
    issues = check_layout(root)
    for issue in issues:
        print(f"{issue.level}: {issue.path.relative_to(root.parent)}: {issue.message}")
    errors = sum(issue.level is Level.ERROR for issue in issues)
    print(f"content_check: {errors} error(s), {len(issues) - errors} warning(s) in {root}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
