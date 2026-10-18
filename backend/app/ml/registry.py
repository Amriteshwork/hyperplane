"""Discovers algorithm packs and exposes them by id."""

import importlib
import pkgutil
from collections.abc import Iterable, Iterator, Sequence
from types import ModuleType

from app.ml import algorithms
from app.ml.base import AlgorithmPack


class PackDiscoveryError(RuntimeError):
    """A pack package is malformed (no `PACK`, wrong type, duplicate id)."""


class UnknownAlgorithmError(KeyError):
    """No pack is registered under this id."""


class AlgorithmRegistry:
    def __init__(self, packs: Iterable[AlgorithmPack] = ()) -> None:
        self._packs: dict[str, AlgorithmPack] = {}
        for pack in packs:
            self.register(pack)

    @classmethod
    def discover(cls, package: ModuleType = algorithms) -> AlgorithmRegistry:
        """Import every sub-package of `package` and register its `PACK`."""
        registry = cls()
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if not info.ispkg:
                continue
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            pack = getattr(module, "PACK", None)
            if pack is None:
                raise PackDiscoveryError(f"{module.__name__} does not export PACK")
            registry.register(pack)
        return registry

    def register(self, pack: object) -> None:
        if not isinstance(pack, AlgorithmPack):
            raise PackDiscoveryError(f"{pack!r} does not implement AlgorithmPack")
        pack_id = pack.spec.id
        if pack_id in self._packs:
            raise PackDiscoveryError(f"Duplicate algorithm id {pack_id!r}")
        self._packs[pack_id] = pack

    def get(self, algorithm_id: str) -> AlgorithmPack:
        try:
            return self._packs[algorithm_id]
        except KeyError:
            raise UnknownAlgorithmError(algorithm_id) from None

    def list(self) -> Sequence[AlgorithmPack]:
        """All packs, ordered by id."""
        return [self._packs[k] for k in sorted(self._packs)]

    def __len__(self) -> int:
        return len(self._packs)

    def __contains__(self, algorithm_id: object) -> bool:
        return algorithm_id in self._packs

    def __iter__(self) -> Iterator[AlgorithmPack]:
        return iter(self.list())
