"""Causal masking in PyTorch: a lower-triangular mask, and the two opposite mask conventions."""

import torch
import torch.nn.functional as F

# @a:reuse
from self_attention_pytorch import attention


# @a:mask
def causal_mask(T):
    """True on and below the diagonal: token t may look at tokens 0..t."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


# @a:demo
if __name__ == "__main__":
    # @a:weights
    torch.manual_seed(0)
    T, d = 5, 4
    X = torch.randn(T, d, dtype=torch.float64)
    W_Q, W_K, W_V = (torch.randn(d, d, dtype=torch.float64) / d**0.5 for _ in range(3))
    out, w = attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask(T))
    print(w.round(decimals=2))
    assert torch.all(w[~causal_mask(T)] == 0)  # nothing above the diagonal

    # @a:builtin
    fused = F.scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V, is_causal=True)
    assert torch.allclose(out, fused)
    print("matches F.scaled_dot_product_attention(..., is_causal=True)")

    # @a:conventions
    heads = torch.nn.MultiheadAttention(d, num_heads=1, bias=False, dtype=torch.float64)
    blocked = ~causal_mask(T)  # nn.MultiheadAttention wants True = "may NOT look"
    _, w_mha = heads(X, X, X, attn_mask=blocked)
    assert torch.all(w_mha[blocked] == 0)
    print("nn.MultiheadAttention: True means blocked; F.scaled_dot_product_attention: allowed")

    # @a:one-pass
    with torch.no_grad():  # one forward pass serves every position at once...
        all_at_once, _ = attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask(T))
        for t in range(T):  # ...and equals running each prefix separately
            prefix, _ = attention(X[: t + 1] @ W_Q, X[: t + 1] @ W_K, X[: t + 1] @ W_V)
            assert torch.allclose(all_at_once[t], prefix[-1])
    print(f"one masked pass = {T} separate predictions")
