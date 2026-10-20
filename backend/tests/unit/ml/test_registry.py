"""Registry discovery, using throwaway pack packages written to a temp dir."""

import importlib
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from app.ml.registry import AlgorithmRegistry, PackDiscoveryError, UnknownAlgorithmError

VALID_PACK = """
from pydantic import BaseModel

from app.ml.base import AlgorithmSpec, Suitability, TaskType


class Params(BaseModel):
    pass


class Pack:
    spec = AlgorithmSpec(
        id="{id}", name="Fake", family="fakes", task_types={{TaskType.REGRESSION}},
        difficulty=1, max_rows_default=1000, content_path="content/algorithms/{id}",
    )
    params_model = Params

    def build_estimator(self, params):
        raise AssertionError("not called in discovery tests")

    def templates(self):
        return {{}}

    def score(self, profile, task):
        return Suitability(score=0.5)


PACK = Pack()
"""

type MakePackage = Callable[[dict[str, str]], ModuleType]


@pytest.fixture
def make_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MakePackage]:
    """Build an importable package whose sub-packages have the given `__init__` sources."""
    monkeypatch.syspath_prepend(str(tmp_path))
    created: list[str] = []

    def make(subpackages: dict[str, str]) -> ModuleType:
        name = f"fake_algorithms_{len(created)}"
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("")
        (root / "not_a_pack.py").write_text("")  # plain modules are ignored
        for sub, source in subpackages.items():
            (root / sub).mkdir()
            (root / sub / "__init__.py").write_text(textwrap.dedent(source))
        created.append(name)
        importlib.invalidate_caches()
        return importlib.import_module(name)

    yield make
    for module in [m for m in sys.modules if m.startswith(tuple(created))]:
        del sys.modules[module]


def test_real_algorithms_package_registers_every_pack() -> None:
    registry = AlgorithmRegistry.discover()
    assert [p.spec.id for p in registry.list()] == [
        "elastic_net",
        "lasso",
        "linear_regression",
        "logistic_regression",
        "polynomial_regression",
        "ridge",
    ]


def test_discovers_packs_sorted_by_id(make_package: MakePackage) -> None:
    package = make_package(
        {"zeta": VALID_PACK.format(id="zeta"), "alpha": VALID_PACK.format(id="alpha")}
    )

    registry = AlgorithmRegistry.discover(package)

    assert [p.spec.id for p in registry.list()] == ["alpha", "zeta"]
    assert registry.get("alpha").spec.name == "Fake"
    assert "zeta" in registry


def test_unknown_id_raises(make_package: MakePackage) -> None:
    registry = AlgorithmRegistry.discover(make_package({}))
    with pytest.raises(UnknownAlgorithmError):
        registry.get("nope")


def test_subpackage_without_pack_is_an_error(make_package: MakePackage) -> None:
    package = make_package({"broken": "X = 1\n"})
    with pytest.raises(PackDiscoveryError, match="does not export PACK"):
        AlgorithmRegistry.discover(package)


def test_object_missing_capabilities_is_rejected(make_package: MakePackage) -> None:
    package = make_package({"partial": "PACK = object()\n"})
    with pytest.raises(PackDiscoveryError, match="does not implement AlgorithmPack"):
        AlgorithmRegistry.discover(package)


def test_duplicate_ids_are_rejected(make_package: MakePackage) -> None:
    package = make_package({"a": VALID_PACK.format(id="same"), "b": VALID_PACK.format(id="same")})
    with pytest.raises(PackDiscoveryError, match="Duplicate algorithm id 'same'"):
        AlgorithmRegistry.discover(package)
