"""Distance metrics in NumPy: Euclidean, Manhattan, Chebyshev and cosine, from their definitions."""

import numpy as np

# @a:reuse
from vectors_numpy import dot, norm


# @a:minkowski
def minkowski(u, v, p):
    """(Σ |uᵢ − vᵢ|ᵖ)^(1/p): p = 1 is Manhattan, p = 2 is Euclidean, p → ∞ is Chebyshev."""
    return np.sum(np.abs(u - v) ** p) ** (1 / p)


# @a:named
def manhattan(u, v):
    return np.sum(np.abs(u - v))


def euclidean(u, v):
    return norm(u - v)


def chebyshev(u, v):
    return np.max(np.abs(u - v))


# @a:cosine
def cosine_distance(u, v):
    """1 − cos θ: 0 when two vectors point the same way, whatever their lengths."""
    return 1 - dot(u, v) / (norm(u) * norm(v))


# @a:pairwise
def pairwise(A, B):
    """Euclidean distance from every row of A to every row of B, in one broadcast."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


# @a:demo
if __name__ == "__main__":
    # @a:triangle
    u, v = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    print(f"manhattan {manhattan(u, v)}, euclidean {euclidean(u, v)}, chebyshev {chebyshev(u, v)}")
    assert (manhattan(u, v), euclidean(u, v), chebyshev(u, v)) == (7.0, 5.0, 4.0)
    assert np.isclose(minkowski(u, v, 1), 7.0) and np.isclose(minkowski(u, v, 2), 5.0)
    assert np.isclose(minkowski(u, v, 100), 4.0, atol=0.01)  # large p approaches the maximum

    # @a:disagree
    query = np.array([0.0, 0.0])
    even, one_sided = np.array([2.0, 2.0]), np.array([3.0, 0.0])
    print(f"even: manhattan {manhattan(query, even)}, euclidean {euclidean(query, even):.2f}")
    print(f"one-sided: manhattan {manhattan(query, one_sided)}, euclidean {euclidean(query, one_sided)}")
    assert manhattan(query, one_sided) < manhattan(query, even)  # Manhattan picks one-sided ...
    assert euclidean(query, even) < euclidean(query, one_sided)  # ... Euclidean picks even

    # @a:cosine-demo
    a, longer, sideways = np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.array([-2.0, 1.0])
    print(f"cosine: same direction {cosine_distance(a, longer):.3f}, at 90° {cosine_distance(a, sideways):.3f}")
    assert np.isclose(cosine_distance(a, longer), 0.0)  # far apart, yet distance 0
    assert np.isclose(cosine_distance(a, sideways), 1.0)

    # @a:scale
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 70, size=200)  # years
    income = rng.uniform(20_000, 120_000, size=200)  # dollars
    X = np.column_stack([age, income])
    query = X[:1]
    raw = np.argsort(pairwise(query, X)[0])[1:6]  # the 5 nearest, skipping the row itself
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    scaled = np.argsort(pairwise(Z[:1], Z)[0])[1:6]
    age_gap_raw = np.abs(age[raw] - age[0]).mean()
    age_gap_scaled = np.abs(age[scaled] - age[0]).mean()
    print(f"mean age gap to the 5 nearest: {age_gap_raw:.1f} years raw, {age_gap_scaled:.1f} scaled")
    assert age_gap_scaled < age_gap_raw  # unscaled, income alone chose the neighbours

    # @a:checks
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    looped = [[euclidean(p, q) for q in B] for p in A]
    assert np.allclose(pairwise(A, B), looped)
    p, q, r = rng.normal(size=(3, 10))
    assert euclidean(p, r) <= euclidean(p, q) + euclidean(q, r)  # the triangle inequality
