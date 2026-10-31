"""Cross-attention in NumPy: one sequence asks the questions, another supplies the answers."""

import numpy as np

# @a:reuse
from multi_head_attention_numpy import multi_head_attention
from self_attention_numpy import attention


# @a:cross
def cross_attention(Y, X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """Queries from Y (T, d), keys and values from X (S, d): every token of Y reads all of X."""
    Q, K, V = Y @ W_Q, X @ W_K, X @ W_V  # the only change from self-attention: K and V come from X
    size = Q.shape[1] // n_heads
    heads = [attention(Q[:, s], K[:, s], V[:, s], mask)
             for s in (slice(h * size, (h + 1) * size) for h in range(n_heads))]
    outputs, weights = zip(*heads)
    return np.concatenate(outputs, axis=1) @ W_O, list(weights)  # (T, d) and h × (T, S)


# @a:demo
if __name__ == "__main__":
    # @a:shapes
    rng = np.random.default_rng(0)
    T, S, d, h = 4, 6, 16, 4  # 4 target tokens read 6 source tokens
    Y, X = rng.normal(size=(T, d)), rng.normal(size=(S, d))
    W_Q, W_K, W_V, W_O = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
    out, weights = cross_attention(Y, X, W_Q, W_K, W_V, W_O, h)
    print(f"target {Y.shape} reads source {X.shape}: output {out.shape}, weights {weights[0].shape}")
    assert out.shape == Y.shape and weights[0].shape == (T, S)
    assert np.allclose(weights[0].sum(axis=1), 1)  # each target token spreads 100% over the source

    # @a:self
    same, _ = cross_attention(X, X, W_Q, W_K, W_V, W_O, h)
    assert np.allclose(same, multi_head_attention(X, W_Q, W_K, W_V, W_O, h)[0])
    print("cross-attention of a sequence with itself is self-attention")

    # @a:any-length
    longer = np.vstack([X, rng.normal(size=(5, d))])  # a source sentence 5 tokens longer
    out_long, w_long = cross_attention(Y, longer, W_Q, W_K, W_V, W_O, h)
    assert out_long.shape == Y.shape and w_long[0].shape == (T, S + 5)
    print(f"source of {S + 5} tokens: output still {out_long.shape}, the target's shape")

    # @a:align
    english, french = ["the", "black", "cat", "sleeps"], ["le", "chat", "noir", "dort"]
    meaning = np.eye(4)  # source word j is just "meaning j"
    asks = [english.index(w) for w in ("the", "cat", "black", "sleeps")]  # what each French word needs
    _, (w,) = cross_attention(meaning[asks], meaning, 10 * np.eye(4), np.eye(4), np.eye(4), np.eye(4), 1)
    for word, row in zip(french, w):
        print(f"  {word:>5} reads {english[row.argmax()]!r} ({row.max():.0%})")
    assert [english[j] for j in w.argmax(axis=1)] == ["the", "cat", "black", "sleeps"]  # order swapped

    # @a:one-by-one
    rows = [cross_attention(Y[t : t + 1], X, W_Q, W_K, W_V, W_O, h)[0] for t in range(T)]
    assert np.allclose(np.vstack(rows), out)  # target tokens never see each other here
    print("each target token's read is independent: one at a time gives the same answer")
