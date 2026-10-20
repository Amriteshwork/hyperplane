"""Matrix multiplication in NumPy: built from dot products of rows and columns."""

import numpy as np

# @a:reuse
from vectors_numpy import dot


# @a:matmul
def matmul(A, B):
    """(AB)ᵢⱼ = row i of A · column j of B."""
    assert A.shape[1] == B.shape[0], f"inner dimensions differ: {A.shape} @ {B.shape}"
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = dot(A[i, :], B[:, j])
    return C


# @a:demo
if __name__ == "__main__":
    # @a:example
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 × 2
    B = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])  # 2 × 3
    C = matmul(A, B)  # 3 × 3
    print("A @ B =\n", C)
    assert C[0, 0] == 1 * 1 + 2 * 2  # row 0 of A · column 0 of B
    assert np.allclose(C, A @ B)  # NumPy's `@` computes the same, much faster

    # @a:predictions
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))  # 5 examples, 3 features
    w = np.array([[2.0], [-1.0], [0.5]])  # weights as a 3 × 1 column
    assert np.allclose(matmul(X, w)[:, 0], [dot(row, w[:, 0]) for row in X])

    # @a:rules
    assert matmul(A, B).shape == (3, 3) and matmul(B, A).shape == (2, 2)  # order matters
    assert np.allclose(matmul(A, B).T, matmul(B.T, A.T))  # (AB)ᵀ = BᵀAᵀ
    assert np.allclose(matmul(A, np.eye(2)), A)  # the identity changes nothing

    # @a:gram
    G = matmul(X.T, X)  # XᵀX: the matrix in the normal equation, 3 × 3
    assert np.allclose(G, G.T)  # always symmetric
    assert np.isclose(G[1, 1], dot(X[:, 1], X[:, 1]))  # its diagonal: each feature's squared length
