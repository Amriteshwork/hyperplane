"""Vectors in PyTorch: the dot product and the norm, written from their definitions."""

import math

import torch


# @a:dot
def dot(u, v):
    """u · v = Σ uᵢ vᵢ: multiply matching entries, add them up."""
    return (u * v).sum()


# @a:norm
def norm(v):
    """‖v‖ = √(v · v): Pythagoras in any number of dimensions."""
    return dot(v, v).sqrt()


# @a:demo
if __name__ == "__main__":
    # @a:examples
    u = torch.tensor([3.0, 4.0])
    v = torch.tensor([-4.0, 3.0])  # u rotated by 90°
    print(f"‖u‖ = {norm(u)},  u · v = {dot(u, v)},  u · u = {dot(u, u)}")
    assert norm(u) == 5.0  # the 3-4-5 triangle
    assert dot(u, v) == 0.0  # perpendicular vectors

    # @a:angle
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])
    cos_theta = dot(a, b) / (norm(a) * norm(b))
    angle = math.degrees(torch.arccos(cos_theta).item())
    print(f"angle between a and b: {angle:.1f}°")
    assert abs(angle - 45.0) < 1e-4

    # @a:prediction
    x = torch.tensor([2.0, 1.0, 0.5])  # one example's features
    w = torch.tensor([1.5, -2.0, 4.0])  # a linear model's weights
    print("prediction w · x =", dot(w, x).item())
    assert dot(w, x) == 3.0

    # @a:checks
    torch.manual_seed(0)
    p, q = torch.randn(100, dtype=torch.float64), torch.randn(100, dtype=torch.float64)
    assert torch.isclose(dot(p, q), torch.dot(p, q))  # matches the built-in
    assert torch.isclose(norm(p), torch.linalg.norm(p))
    assert abs(dot(p, q)) <= norm(p) * norm(q)  # Cauchy–Schwarz: |cos θ| ≤ 1
