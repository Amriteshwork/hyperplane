"""Positional encoding in PyTorch: added sinusoids, learned positions, and rotary embeddings (RoPE)."""

import torch


# @a:sinusoidal
def sinusoidal(T, d):
    """PE[t, 2i] = sin(t·ωᵢ), PE[t, 2i+1] = cos(t·ωᵢ), with ωᵢ = 1 / 10000^(2i/d)."""
    angle = torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)
    PE = torch.zeros(T, d)
    PE[:, 0::2], PE[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return PE


# @a:rope
def rope(X):
    """Rotate each pair of columns of row t by angle t·ωᵢ (applied to queries and keys)."""
    T, d = X.shape[-2:]
    angle = torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = X[..., 0::2], X[..., 1::2]
    out = torch.empty_like(X)
    out[..., 0::2], out[..., 1::2] = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    return out


# @a:demo
if __name__ == "__main__":
    # @a:learned
    torch.manual_seed(0)
    T, d = 32, 16
    positions = torch.nn.Embedding(T, d)  # one learnable vector per position, like a token
    tokens = torch.randn(T, d)
    X = tokens + positions(torch.arange(T))  # what GPT-2 does: add, don't concatenate
    assert X.shape == (T, d)
    assert torch.allclose(sinusoidal(T, d)[:, 0], torch.sin(torch.arange(T).float()))  # ω₀ = 1

    # @a:relative
    q, k = torch.randn(d), torch.randn(d)

    def score(t, s):
        """Query q placed at position t against key k placed at position s, both rotated."""
        Q, K = rope(q.expand(T, d)), rope(k.expand(T, d))
        return (Q[t] @ K[s]).item()

    print(f"score(5, 2) {score(5, 2):.4f}   score(20, 17) {score(20, 17):.4f}   "
          f"score(5, 4) {score(5, 4):.4f}")
    assert abs(score(5, 2) - score(20, 17)) < 1e-4  # only the distance t − s matters
    assert abs(score(5, 2) - score(5, 4)) > 1e-3  # a different distance, a different score
    assert torch.allclose(rope(tokens).norm(dim=1), tokens.norm(dim=1))  # rotations keep lengths
