import ast
import textwrap

import pytest

from app.ml.base import Annotation, ExplainedTemplate
from app.ml.codegen import AnchorMismatchError, CodegenContext, render, strip_anchors


def note(span: str = "line") -> Annotation:
    return Annotation(short="s", deep="d", span=span)  # type: ignore[arg-type]


def spans(source: str, annotations: dict[str, Annotation]) -> dict[str, tuple[int, int]]:
    rendered = strip_anchors(textwrap.dedent(source), annotations, filename="x.py")
    return {s.anchor: (s.start_line, s.end_line) for s in rendered.spans}


def test_own_line_anchor_explains_next_code_line_and_is_removed() -> None:
    source = "\n# @a:one\n\nx = 1\ny = 2\n"
    rendered = strip_anchors(source, {"one": note()}, filename="x.py")
    assert rendered.code == "\nx = 1\ny = 2\n".lstrip("\n")
    assert "@a:" not in rendered.code
    # Leading blank lines are dropped; the blank line between anchor and code is kept.
    assert [(s.start_line, s.end_line) for s in rendered.spans] == [(1, 1)]


def test_trailing_anchor_explains_its_own_line() -> None:
    rendered = strip_anchors("a = 1  # @a:one\nb = 2\n", {"one": note()}, filename="x.py")
    assert rendered.code == "a = 1\nb = 2\n"
    assert rendered.spans[0].start_line == 1


def test_block_runs_to_next_anchor_without_trailing_blanks() -> None:
    source = """
        # @a:first
        a = 1
        b = 2

        # @a:second
        c = 3
        d = 4  # @a:third
    """
    result = spans(source, {"first": note("block"), "second": note("block"), "third": note()})
    assert result == {"first": (1, 2), "second": (4, 4), "third": (5, 5)}


@pytest.mark.parametrize(
    ("source", "annotations"),
    [
        ("x = 1  # @a:missing\n", {}),
        ("x = 1\n", {"orphan": note()}),
        ("x = 1  # @a:dup\ny = 2  # @a:dup\n", {"dup": note()}),
        ("x = 1\n# @a:dangling\n", {"dangling": note()}),
    ],
)
def test_anchor_mismatches_fail(source: str, annotations: dict[str, Annotation]) -> None:
    with pytest.raises(AnchorMismatchError):
        strip_anchors(source, annotations, filename="x.py")


def context(**overrides: object) -> CodegenContext:
    values: dict[str, object] = {
        "data_file": "data.csv",
        "target": "y",
        "task_type": "regression",
        "numeric_features": ["a"],
        "test_size": 0.2,
        "seed": 1,
        "params": {},
    }
    return CodegenContext.model_validate(values | overrides)


def test_user_values_become_python_literals_not_code() -> None:
    template = ExplainedTemplate(
        source="TARGET = {{ target | py }}\nCOLS = {{ numeric_features | py }}\n",
        templated=True,
    )
    hostile = 'x"); import os; os.system("rm -rf /"); ("'
    rendered = render(template, context(target=hostile, numeric_features=["it's", "a\nb"]))

    tree = ast.parse(rendered.code)
    assert len(tree.body) == 2  # still two assignments: nothing was injected
    assert ast.literal_eval(tree.body[0].value) == hostile  # type: ignore[attr-defined]
    assert ast.literal_eval(tree.body[1].value) == ["it's", "a\nb"]  # type: ignore[attr-defined]


def test_templates_need_a_context_and_plain_sources_are_not_rendered() -> None:
    with pytest.raises(ValueError, match="needs a CodegenContext"):
        render(ExplainedTemplate(source="{{ x }}", templated=True))
    assert render(ExplainedTemplate(source="d = {'a': 1}\n")).code == "d = {'a': 1}\n"
