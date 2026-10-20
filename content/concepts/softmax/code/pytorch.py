"""Softmax in PyTorch: turn one score per class into probabilities that add up to 1."""

import torch

# @a:reuse
from logarithms_exponentials_pytorch import logsumexp
from sigmoid_pytorch import sigmoid


# @a:softmax
def softmax(Z):
    """Row-wise e^z / Σ e^z, via log-softmax so that no exponent overflows."""
    return torch.exp(Z - logsumexp(Z, dim=1)[:, None])


# @a:demo
if __name__ == "__main__":
    f64 = torch.float64

    # @a:scores
    Z = torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0], [1000.0, 1001.0, 999.0]], dtype=f64)
    P = softmax(Z)
    print(P.round(decimals=3))

    # @a:checks
    assert torch.allclose(P.sum(dim=1), torch.ones(3, dtype=f64)) and torch.all(P > 0)
    assert torch.equal(P.argmax(dim=1), Z.argmax(dim=1))  # biggest score, biggest share
    assert torch.allclose(P[1], torch.full((3,), 1 / 3, dtype=f64))  # equal scores, equal chances
    assert torch.allclose(softmax(Z + 5.0), P)  # only differences between scores matter
    assert torch.allclose(P, torch.softmax(Z, dim=1))  # PyTorch's built-in agrees

    # @a:two-classes
    z = torch.linspace(-4, 4, 9, dtype=f64)
    two = softmax(torch.stack([torch.zeros_like(z), z], dim=1))
    assert torch.allclose(two[:, 1], sigmoid(z))  # two classes: softmax is the sigmoid
