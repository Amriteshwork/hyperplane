"""Cross-attention in PyTorch: batched over heads and sentences, checked against nn.MultiheadAttention."""

import torch

# @a:reuse
from multi_head_attention_pytorch import multi_head_attention


# @a:cross
def cross_attention(Y, X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """Y (…, T, d) asks, X (…, S, d) answers: → (…, T, d) and weights (…, heads, T, S)."""
    *batch, T, d = Y.shape
    S = X.shape[-2]  # the source may be any length

    def split(M, length):  # column block h of M becomes head h
        return M.view(*batch, length, n_heads, d // n_heads).transpose(-3, -2)

    Q, K, V = split(Y @ W_Q, T), split(X @ W_K, S), split(X @ W_V, S)
    scores = Q @ K.transpose(-1, -2) / (d // n_heads) ** 0.5  # (…, heads, T, S)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ V).transpose(-3, -2).reshape(*batch, T, d) @ W_O, weights


# @a:demo
if __name__ == "__main__":
    # @a:builtin
    torch.manual_seed(0)
    T, S, d, h = 4, 6, 16, 4
    Y, X = torch.randn(T, d, dtype=torch.float64), torch.randn(S, d, dtype=torch.float64)
    W_Q, W_K, W_V, W_O = (torch.randn(d, d, dtype=torch.float64) / d**0.5 for _ in range(4))
    out, weights = cross_attention(Y, X, W_Q, W_K, W_V, W_O, h)
    layer = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        layer.in_proj_weight.copy_(torch.cat([W_Q.T, W_K.T, W_V.T]))  # PyTorch stores Wᵀ
        layer.out_proj.weight.copy_(W_O.T)
        theirs, _ = layer(Y[None], X[None], X[None], need_weights=False)  # query, key, value
    assert torch.allclose(out, theirs[0])
    print(f"weights {tuple(weights.shape)} = (heads, target, source); matches nn.MultiheadAttention")

    # @a:self
    same, _ = cross_attention(X, X, W_Q, W_K, W_V, W_O, h)
    assert torch.allclose(same, multi_head_attention(X, W_Q, W_K, W_V, W_O, h)[0])

    # @a:padding
    padded = torch.cat([X, torch.zeros(3, d, dtype=torch.float64)])  # 3 filler rows after the source
    real = torch.arange(S + 3) < S  # (S + 3,): which source positions are real
    out_pad, w_pad = cross_attention(Y, padded, W_Q, W_K, W_V, W_O, h, real.expand(T, S + 3))
    assert torch.allclose(out_pad, out) and w_pad[..., S:].sum() == 0
    print("a padded source with its padding masked gives exactly the same output")

    # @a:batch
    Ys, Xs = torch.randn(3, T, d, dtype=torch.float64), torch.randn(3, S, d, dtype=torch.float64)
    together, _ = cross_attention(Ys, Xs, W_Q, W_K, W_V, W_O, h)
    for i in range(3):  # sentence i only ever reads source i
        assert torch.allclose(together[i], cross_attention(Ys[i], Xs[i], W_Q, W_K, W_V, W_O, h)[0])
