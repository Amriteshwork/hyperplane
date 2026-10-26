"""Prerequisite graph and imports between concept code (ADR 0016)."""

import pytest

from app.content.code_imports import code_uses
from app.content.graph import (
    ConceptGraphError,
    check_graph,
    check_route,
    learning_path,
    reading_order,
)
from app.content.models import CodeLibrary, CodeUse

NUMPY, PYTORCH = CodeLibrary.NUMPY, CodeLibrary.PYTORCH
GRAPH = {
    "vectors": (),
    "matrices": ("vectors",),
    "derivatives": (),
    "loss": (),
    "mse": ("loss",),
    "gradient-descent": ("derivatives", "mse"),
    "neural-network": ("matrices", "gradient-descent"),
}


def test_learning_path_lists_every_prerequisite_after_what_it_needs() -> None:
    path = learning_path(GRAPH, "neural-network")

    assert path == ["vectors", "matrices", "derivatives", "loss", "mse", "gradient-descent"]
    assert learning_path(GRAPH, "vectors") == []


def test_shared_prerequisites_appear_once() -> None:
    graph = {**GRAPH, "attention": ("matrices", "neural-network")}
    path = learning_path(graph, "attention")
    assert path.count("vectors") == 1
    assert path.index("vectors") < path.index("matrices") < path.index("neural-network")


def test_reading_order_includes_the_targets_after_their_prerequisites() -> None:
    order = reading_order(GRAPH, ["mse", "gradient-descent"])

    assert order == ["loss", "mse", "derivatives", "gradient-descent"]
    assert reading_order(GRAPH, ["neural-network"])[-1] == "neural-network"


@pytest.mark.parametrize(
    ("graph", "match"),
    [
        ({"a": ("missing",)}, "unknown prerequisites"),
        ({"a": ("b",), "b": ("c",), "c": ("a",)}, "cycle: a → b → c → a"),
        ({"a": ("a",)}, "cycle"),
    ],
)
def test_broken_graphs_are_rejected(graph: dict[str, tuple[str, ...]], match: str) -> None:
    with pytest.raises(ConceptGraphError, match=match):
        check_graph(graph)


MSE = '''"""MSE."""

import numpy as np


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


SCALE: float = 1.0

if __name__ == "__main__":
    print(mse(np.ones(2), np.zeros(2)))
'''
SOURCES = {"loss": {}, "mse": {NUMPY: MSE, PYTORCH: "import torch\n"}, "other": {NUMPY: MSE}}


def uses(source: str, library: CodeLibrary = NUMPY, ancestors: set[str] | None = None):
    ancestors = {"loss", "mse"} if ancestors is None else ancestors
    return code_uses("x.py", source, library, ancestors, SOURCES)


def test_imports_from_prerequisites_are_recorded() -> None:
    source = (
        "from mean_squared_error_numpy import x\nfrom mse_numpy import mse, SCALE\nimport numpy\n"
    )
    assert uses(source) == (CodeUse(concept="mse", names=("mse", "SCALE")),)
    assert uses("import numpy as np\n") == ()


@pytest.mark.parametrize(
    ("source", "library", "match"),
    [
        ("from other_numpy import mse\n", NUMPY, "not a prerequisite"),
        ("from mse_numpy import nope\n", NUMPY, r"defines no \['nope'\]"),
        ("from mse_numpy import mse\n", PYTORCH, "its own library"),
        ("from loss_numpy import f\n", NUMPY, "has no numpy code"),
        ("import mse_numpy\n", NUMPY, "from <module> import"),
        ("def broken(:\n", NUMPY, "x.py:1"),
    ],
)
def test_bad_imports_are_rejected(source: str, library: CodeLibrary, match: str) -> None:
    with pytest.raises(ConceptGraphError, match=match):
        uses(source, library)


def test_imported_files_must_keep_their_demo_under_main() -> None:
    sources = {"mse": {NUMPY: MSE + "print('demo at import time')\n"}}
    with pytest.raises(ConceptGraphError, match=r"mse_numpy.py:\d+: .*__main__"):
        code_uses("x.py", "from mse_numpy import mse\n", NUMPY, {"mse"}, sources)


def test_a_route_may_reorder_the_path_but_never_put_a_concept_before_its_prerequisites() -> None:
    route = ["loss", "mse", "derivatives", "gradient-descent", "vectors", "matrices"]
    check_route(GRAPH, "neural-network", [*route, "neural-network"])


@pytest.mark.parametrize(
    ("goal", "route", "match"),
    [
        ("nope", ["nope"], "unknown goal"),
        ("mse", ["mse"], "missing \\['loss'\\]"),
        ("mse", ["loss", "vectors", "mse"], "doesn't need: \\['vectors'\\]"),
        ("mse", ["loss", "loss", "mse"], "twice"),
        ("mse", ["mse", "loss"], "'mse' comes before its prerequisites \\['loss'\\]"),
    ],
)
def test_bad_routes_say_what_is_wrong(goal: str, route: list[str], match: str) -> None:
    with pytest.raises(ConceptGraphError, match=match):
        check_route(GRAPH, goal, route)
