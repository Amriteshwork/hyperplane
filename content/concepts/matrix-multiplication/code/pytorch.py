"""Matrix multiplication in PyTorch: built from dot products of rows and columns."""

import torch

# @a:reuse
from vectors_pytorch import dot


# @a:matmul
def matmul(A, B):
    """(AB)ᵢⱼ = row i of A · column j of B."""
    assert A.shape[1] == B.shape[0], f"inner dimensions differ: {A.shape} @ {B.shape}"
    C = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = dot(A[i, :], B[:, j])
    return C


# @a:demo
if __name__ == "__main__":
    # @a:example
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 × 2
    B = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])  # 2 × 3
    C = matmul(A, B)  # 3 × 3
    print("A @ B =\n", C)
    assert C[0, 0] == 1 * 1 + 2 * 2  # row 0 of A · column 0 of B
    assert torch.allclose(C, A @ B)  # PyTorch's `@` computes the same, much faster

    # @a:predictions
    torch.manual_seed(0)
    X = torch.randn(5, 3)  # 5 examples, 3 features
    w = torch.tensor([[2.0], [-1.0], [0.5]])  # weights as a 3 × 1 column
    assert torch.allclose(matmul(X, w)[:, 0], torch.stack([dot(row, w[:, 0]) for row in X]))

    # @a:rules
    assert matmul(A, B).shape == (3, 3) and matmul(B, A).shape == (2, 2)  # order matters
    assert torch.allclose(matmul(A, B).T, matmul(B.T, A.T))  # (AB)ᵀ = BᵀAᵀ
    assert torch.allclose(matmul(A, torch.eye(2)), A)  # the identity changes nothing

    # @a:gram
    G = matmul(X.T, X)  # XᵀX: the matrix in the normal equation, 3 × 3
    assert torch.allclose(G, G.T)  # always symmetric
    assert torch.isclose(G[1, 1], dot(X[:, 1], X[:, 1]))  # diagonal: squared column lengths
