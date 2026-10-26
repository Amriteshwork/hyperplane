"""Multi-head attention in PyTorch: all heads at once by reshaping, checked against nn.MultiheadAttention."""

import torch

# @a:reuse
from causal_masking_pytorch import causal_mask
from self_attention_pytorch import attention


# @a:mha
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """All heads in one batched matmul: (…, T, d) → (…, heads, T, d/heads) → attend → (…, T, d)."""
    *batch, T, d = X.shape  # any leading batch dimensions ride along

    def split(M):  # column block h of M becomes head h
        return M.view(*batch, T, n_heads, d // n_heads).transpose(-3, -2)

    Q, K, V = split(X @ W_Q), split(X @ W_K), split(X @ W_V)
    scores = Q @ K.transpose(-1, -2) / (d // n_heads) ** 0.5  # (…, heads, T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # one mask, broadcast to every head
    weights = torch.softmax(scores, dim=-1)
    return (weights @ V).transpose(-3, -2).reshape(*batch, T, d) @ W_O, weights


# @a:demo
if __name__ == "__main__":
    # @a:loop
    torch.manual_seed(0)
    T, d, h = 6, 16, 4
    X = torch.randn(T, d, dtype=torch.float64)
    W_Q, W_K, W_V, W_O = (torch.randn(d, d, dtype=torch.float64) / d**0.5 for _ in range(4))
    out, weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, h, causal_mask(T))
    cols = [slice(i * d // h, (i + 1) * d // h) for i in range(h)]
    looped = torch.cat([attention((X @ W_Q)[:, c], (X @ W_K)[:, c], (X @ W_V)[:, c],
                                  causal_mask(T))[0] for c in cols], dim=1) @ W_O
    assert torch.allclose(out, looped)  # reshaping = looping over heads, minus the loop
    print(f"weights: {tuple(weights.shape)} = (heads, queries, keys)")

    # @a:builtin
    layer = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        layer.in_proj_weight.copy_(torch.cat([W_Q.T, W_K.T, W_V.T]))  # PyTorch stores Wᵀ
        layer.out_proj.weight.copy_(W_O.T)
        theirs, _ = layer(X[None], X[None], X[None], attn_mask=~causal_mask(T), need_weights=False)
    assert torch.allclose(out, theirs[0])
    print("matches nn.MultiheadAttention with the same weights")

    # @a:batch
    batch = torch.randn(3, T, d, dtype=torch.float64)  # three sequences at once
    together, _ = multi_head_attention(batch, W_Q, W_K, W_V, W_O, h, causal_mask(T))
    for i in range(3):
        alone, _ = multi_head_attention(batch[i], W_Q, W_K, W_V, W_O, h, causal_mask(T))
        assert torch.allclose(together[i], alone)  # sequences never mix
