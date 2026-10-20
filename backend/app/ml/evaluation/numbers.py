"""Keeping plot payloads small: subsample points and round values for JSON."""

import numpy as np
from numpy.typing import NDArray

#: Charts get at most this many points (a fixed subset), to keep payloads small.
MAX_POINTS = 1_000


def subset(n: int) -> NDArray[np.intp]:
    """A deterministic random subset of `range(n)`, in order."""
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, MAX_POINTS, replace=False))


def evenly(n: int) -> NDArray[np.intp]:
    """At most MAX_POINTS evenly spaced indices of `range(n)`, always keeping both ends."""
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, MAX_POINTS).round().astype(np.intp))


def points(x: NDArray[np.float64], y: NDArray[np.float64]) -> list[tuple[float, float]]:
    return [(round6(a), round6(b)) for a, b in zip(x, y, strict=True)]


def round6(value: float) -> float:
    return float(f"{value:.6g}")
