"""Every `::figure[name]` in content/ names a figure the frontend registers; directives stand alone."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
CONTENT = ROOT / "content"
REGISTRY = ROOT / "frontend/src/app/shared/figures/figure-registry.ts"
DIRECTIVE = re.compile(r"^::figure\[([a-z0-9-]+)\]", re.MULTILINE)
#: Figures and Manim videos (ADR 0022) are both block directives.
ANY_DIRECTIVE = re.compile(r"^::(figure|video)\[")
# Keys of DEFAULT_FIGURES: `'name': () => import(...)`.
ENTRY = re.compile(r"^\s*'([a-z0-9-]+)':\s*\(\)\s*=>", re.MULTILINE)


def used_figures() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    for page in sorted(CONTENT.rglob("*.md")):
        for name in DIRECTIVE.findall(page.read_text()):
            used.setdefault(name, []).append(str(page.relative_to(CONTENT)))
    return used


@pytest.mark.skipif(not REGISTRY.exists(), reason="frontend checkout not present")
def test_content_only_uses_registered_figures() -> None:
    registered = set(ENTRY.findall(REGISTRY.read_text()))
    assert registered, "no figures parsed from the registry; has its format changed?"
    unknown = {name: pages for name, pages in used_figures().items() if name not in registered}
    assert not unknown, f"figures used in content but not registered: {unknown}"


def test_figure_directives_are_whole_lines() -> None:
    """A directive inside a paragraph would render as text, so it must stand alone."""
    for page in CONTENT.rglob("*.md"):
        lines = page.read_text().splitlines()
        for i, line in enumerate(lines):
            if ANY_DIRECTIVE.match(line):
                assert i == 0 or not lines[i - 1].strip(), (
                    f"{page}:{i + 1} needs a blank line before"
                )
