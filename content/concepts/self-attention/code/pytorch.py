"""Self-attention in PyTorch: softmax(QKᵀ/√d)·V, checked against PyTorch's own kernel."""

import torch
import torch.nn.functional as F

# @a:reuse
from softmax_pytorch import softmax


# @a:attention
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention; returns the outputs and the weights, so we can look at them."""
    scores = Q @ K.T / K.shape[1] ** 0.5  # (T, T): every query against every key
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # masked pairs get weight exactly 0
    weights = softmax(scores)  # each row sums to 1
    return weights @ V, weights


# @a:demo
if __name__ == "__main__":
    # @a:projections
    torch.manual_seed(0)
    T, d, d_k = 5, 8, 4
    X = torch.randn(T, d, dtype=torch.float64)  # five token embeddings
    W_Q, W_K, W_V = (torch.randn(d, d_k, dtype=torch.float64) / d**0.5 for _ in range(3))
    out, w = attention(X @ W_Q, X @ W_K, X @ W_V)
    assert out.shape == (T, d_k) and torch.allclose(w.sum(dim=1), torch.ones(T, dtype=w.dtype))

    # @a:builtin
    fused = F.scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V)
    assert torch.allclose(out, fused)  # the same maths as PyTorch's optimised kernel
    print("matches F.scaled_dot_product_attention")

    # @a:gradients
    d_k = 512
    Q, K = torch.randn(2, 16, d_k, dtype=torch.float64)
    for scale in (d_k**0.5, 1.0):  # with and without the √d
        scores = (Q @ K.T / scale).requires_grad_()
        weights = torch.softmax(scores, dim=1)
        (weights * torch.randn(16, 16, dtype=torch.float64)).sum().backward()  # any loss will do
        top = weights.max(dim=1).values.median().item()
        grad = scores.grad.norm(dim=1).median().item()  # how much each row's scores can learn
        print(f"divide by {scale:4.1f}: typical top weight {top:.3f}, score gradient {grad:.1e}")
    assert top > 0.99 and grad < 1e-3  # unscaled: softmax saturates and its gradient vanishes
