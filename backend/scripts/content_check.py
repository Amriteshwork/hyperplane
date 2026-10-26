"""Validate content/ (ARCHITECTURE §8) and report the content backlog.

Errors (exit 1): bad layout, invalid frontmatter/resources, unknown resource ids, missing overview
sections, packs without content, unknown concepts (packs, code annotations) and
code anchors without annotations. Warnings: unreferenced resources and curation gaps (the backlog).

Usage: uv run python -m scripts.content_check [--content-dir PATH] [--backlog FILE.md]
"""

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.content.catalog import ContentCatalog, ContentError, UnknownContentError
from app.content.gaps import content_gaps, missing_sections
from app.core.config import get_settings
from app.ml.registry import AlgorithmRegistry

SECTIONS = ("algorithms", "concepts", "families", "resources")
OPTIONAL_SECTIONS = ("tracks", "papers")
REQUIRED_FILES = {
    "algorithms": ("overview.md", "resources.yaml"),
    "concepts": ("concept.md", "resources.yaml"),
    "tracks": ("track.md", "resources.yaml"),
    "papers": ("paper.md", "resources.yaml"),
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
        if entry.name not in IGNORED and entry.name not in SECTIONS + OPTIONAL_SECTIONS:
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

    resources = root / "resources"
    if resources.is_dir():
        for item in sorted(p for p in resources.iterdir() if p.name not in IGNORED):
            if not (item.is_file() and item.suffix == ".yaml"):
                issues.append(Issue(Level.ERROR, item, "resources/ holds <resource_id>.yaml files"))

    return issues


def check_content(root: Path, registry: AlgorithmRegistry) -> list[Issue]:
    """Schema, required sections, pack ↔ content links, and curation gaps."""
    try:
        catalog = ContentCatalog.load(root)
    except ContentError as exc:
        return [Issue(Level.ERROR, root, str(exc))]

    issues: list[Issue] = []
    pack_ids = []
    for pack in registry:
        folder = root / "algorithms" / pack.spec.id
        try:
            content = catalog.algorithm(pack.spec.id)
        except UnknownContentError:
            issues.append(Issue(Level.ERROR, folder, "registered pack has no content folder"))
            continue
        pack_ids.append(pack.spec.id)
        for section in missing_sections(content.body):
            issues.append(Issue(Level.ERROR, folder / "overview.md", f"missing '## {section}'"))
        for concept_id in pack.spec.concepts:
            if not catalog.has_concept(concept_id):
                issues.append(Issue(Level.ERROR, folder, f"spec concept {concept_id!r} not found"))
        if catalog.family_of(pack.spec.id) is None:
            issues.append(Issue(Level.ERROR, folder, "not listed in any families/*.md"))

    # Unknown prerequisites, cycles and bad code imports already fail loading (ADR 0016).
    for concept in catalog.concepts():
        for code in concept.code:
            for anchor, annotation in code.annotations.items():
                for linked in annotation.concepts:
                    if not catalog.has_concept(linked):
                        issues.append(
                            Issue(
                                Level.ERROR,
                                root / "concepts" / concept.id / "code" / "annotations.yaml",
                                f"{code.library}.{anchor}: unknown concept {linked!r}",
                            )
                        )

    used = {link.resource.id for a in catalog.algorithms() for link in a.resources}
    used |= {link.resource.id for c in catalog.concepts() for link in c.resources}
    used |= {link.resource.id for t in catalog.tracks() for link in t.resources}
    used |= {link.resource.id for p in catalog.papers() for link in p.resources}
    used |= {p.source.id for p in catalog.papers()}
    for resource in catalog.resources():
        if resource.id not in used:
            path = root / "resources" / f"{resource.id}.yaml"
            issues.append(
                Issue(
                    Level.WARNING, path, "not referenced by any algorithm, concept, track or paper"
                )
            )

    for gap in content_gaps(catalog, pack_ids):
        issues.append(Issue(Level.WARNING, root / gap.item, f"backlog: {gap.message}"))
    return issues


def write_backlog(issues: list[Issue], root: Path, path: Path) -> None:
    lines = ["# Content backlog", "", "Generated by `scripts.content_check`; do not edit.", ""]
    for issue in issues:
        if issue.message.startswith("backlog: "):
            item = issue.path.relative_to(root).as_posix()
            lines.append(f"- [ ] `{item}`: {issue.message.removeprefix('backlog: ')}")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def main(argv: list[str] | None = None, registry: AlgorithmRegistry | None = None) -> int:
    """Print issues; exit 1 if any is an error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--content-dir", type=Path, default=get_settings().content_dir)
    parser.add_argument("--backlog", type=Path, help="also write the backlog as Markdown")
    args = parser.parse_args(argv)

    root: Path = args.content_dir.resolve()
    issues = check_layout(root)
    if not any(issue.level is Level.ERROR for issue in issues):
        issues += check_content(
            root, AlgorithmRegistry.discover() if registry is None else registry
        )
    if args.backlog:
        write_backlog(issues, root, args.backlog)
    for issue in issues:
        print(f"{issue.level}: {issue.path.relative_to(root.parent)}: {issue.message}")
    errors = sum(issue.level is Level.ERROR for issue in issues)
    print(f"content_check: {errors} error(s), {len(issues) - errors} warning(s) in {root}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
