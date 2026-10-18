"""Anchor stripping (ADR 0006). Anchors mark *where* an explanation starts; spans are derived.

- An anchor alone on a line (`# @a:split`) explains the next non-blank line.
- An anchor after code (`x = 1  # @a:split`) explains that line.
- `span: block` extends the explanation up to the next anchor, minus trailing blank lines.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.ml.base import Annotation

_OWN_LINE = re.compile(r"^\s*#\s*@a:(?P<id>[a-z0-9_-]+)\s*$")
_TRAILING = re.compile(r"^(?P<code>.*?\S)\s+#\s*@a:(?P<id>[a-z0-9_-]+)\s*$")


class AnchorMismatchError(ValueError):
    """Anchors and annotations don't match 1:1 (the contract tests catch this first)."""


@dataclass(frozen=True, slots=True)
class ExplainedSpan:
    anchor: str
    #: 1-based, inclusive, in the displayed (anchor-free) code.
    start_line: int
    end_line: int
    annotation: Annotation


@dataclass(frozen=True, slots=True)
class RenderedCode:
    filename: str
    code: str
    spans: tuple[ExplainedSpan, ...]


def find_anchors(source: str) -> list[str]:
    """Anchor ids in order of appearance."""
    ids = []
    for line in source.splitlines():
        match = _OWN_LINE.match(line) or _TRAILING.match(line)
        if match:
            ids.append(match["id"])
    return ids


def strip_anchors(
    source: str, annotations: Mapping[str, Annotation], *, filename: str
) -> RenderedCode:
    lines: list[str] = []
    starts: list[tuple[str, int]] = []  # (anchor, 0-based line in `lines`)
    pending: list[str] = []
    for raw in source.splitlines():
        if own := _OWN_LINE.match(raw):
            pending.append(own["id"])
            continue
        if not lines and not raw.strip():
            continue  # drop leading blank lines (Jinja blocks often leave some)
        trailing = _TRAILING.match(raw)
        lines.append(trailing["code"] if trailing else raw.rstrip())
        if pending and lines[-1].strip():
            starts.extend((anchor, len(lines) - 1) for anchor in pending)
            pending = []
        if trailing:
            starts.append((trailing["id"], len(lines) - 1))
    if pending:
        raise AnchorMismatchError(f"{filename}: anchors with no code after them: {pending}")

    _check_one_to_one(filename, [a for a, _ in starts], annotations)
    spans = tuple(
        _span(anchor, start, starts, lines, annotations[anchor]) for anchor, start in starts
    )
    code = "\n".join(lines).rstrip("\n") + "\n"
    return RenderedCode(filename=filename, code=code, spans=spans)


def _span(
    anchor: str,
    start: int,
    starts: list[tuple[str, int]],
    lines: list[str],
    annotation: Annotation,
) -> ExplainedSpan:
    end = start
    if annotation.span == "block":
        following = [s for _, s in starts if s > start]
        end = (min(following) if following else len(lines)) - 1
        while end > start and not lines[end].strip():
            end -= 1
    return ExplainedSpan(anchor, start + 1, end + 1, annotation)


def _check_one_to_one(
    filename: str, anchors: list[str], annotations: Mapping[str, Annotation]
) -> None:
    duplicates = sorted({a for a in anchors if anchors.count(a) > 1})
    missing = sorted(set(anchors) - set(annotations))
    orphaned = sorted(set(annotations) - set(anchors))
    if duplicates or missing or orphaned:
        raise AnchorMismatchError(
            f"{filename}: duplicate anchors {duplicates}, anchors without annotations "
            f"{missing}, annotations without anchors {orphaned}"
        )
