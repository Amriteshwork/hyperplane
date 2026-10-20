"""Vectors in NumPy: the dot product and the norm, written from their definitions."""

import numpy as np


# @a:dot
def dot(u, v):
    """u · v = Σ uᵢ vᵢ: multiply matching entries, add them up."""
    return np.sum(u * v)


# @a:norm
def norm(v):
    """‖v‖ = √(v · v): Pythagoras in any number of dimensions."""
    return np.sqrt(dot(v, v))


# @a:demo
if __name__ == "__main__":
    # @a:examples
    u = np.array([3.0, 4.0])
    v = np.array([-4.0, 3.0])  # u rotated by 90°
    print(f"‖u‖ = {norm(u)},  u · v = {dot(u, v)},  u · u = {dot(u, u)}")
    assert norm(u) == 5.0  # the 3-4-5 triangle
    assert dot(u, v) == 0.0  # perpendicular vectors

    # @a:angle
    a, b = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    cos_theta = dot(a, b) / (norm(a) * norm(b))
    print(f"angle between a and b: {np.degrees(np.arccos(cos_theta)):.1f}°")
    assert np.isclose(np.degrees(np.arccos(cos_theta)), 45.0)

    # @a:prediction
    x = np.array([2.0, 1.0, 0.5])  # one example's features
    w = np.array([1.5, -2.0, 4.0])  # a linear model's weights
    print("prediction w · x =", dot(w, x))
    assert dot(w, x) == 3.0

    # @a:checks
    rng = np.random.default_rng(0)
    p, q = rng.normal(size=100), rng.normal(size=100)
    assert np.isclose(dot(p, q), np.dot(p, q))  # matches NumPy's built-in
    assert np.isclose(norm(p), np.linalg.norm(p))
    assert abs(dot(p, q)) <= norm(p) * norm(q)  # Cauchy–Schwarz: |cos θ| ≤ 1
