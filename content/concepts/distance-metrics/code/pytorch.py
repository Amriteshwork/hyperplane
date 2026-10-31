"""Distance metrics in PyTorch: `torch.cdist` is the batched version of the definitions."""

import torch


# @a:pairwise
def pairwise(A, B, p=2.0):
    """The Minkowski distance from every row of A to every row of B, in one broadcast."""
    return (A[:, None, :] - B[None, :, :]).abs().pow(p).sum(dim=2).pow(1 / p)


# @a:cosine
def cosine_distance(A, B):
    """1 − cos θ between every row of A and every row of B."""
    unit_A = torch.nn.functional.normalize(A, dim=1)
    unit_B = torch.nn.functional.normalize(B, dim=1)
    return 1 - unit_A @ unit_B.T


# @a:demo
if __name__ == "__main__":
    # @a:checks
    torch.manual_seed(0)
    A, B = torch.randn(5, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    for p in (1.0, 2.0, 3.0):
        assert torch.allclose(pairwise(A, B, p), torch.cdist(A, B, p=p))
    chebyshev = torch.cdist(A, B, p=float("inf"))
    assert torch.allclose(pairwise(A, B, 200.0), chebyshev, atol=0.02)  # large p approaches the max

    # @a:cosine-demo
    similarity = torch.nn.functional.cosine_similarity(A[:, None, :], B[None, :, :], dim=2)
    assert torch.allclose(cosine_distance(A, B), 1 - similarity)
    print("mean Euclidean distance:", round(pairwise(A, B).mean().item(), 3))
    print("mean cosine distance:", round(cosine_distance(A, B).mean().item(), 3))
