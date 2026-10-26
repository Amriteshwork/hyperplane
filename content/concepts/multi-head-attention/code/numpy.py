"""Multi-head attention in NumPy: several small attentions side by side, concatenated and mixed."""

import numpy as np

# @a:reuse
from causal_masking_numpy import causal_mask
from self_attention_numpy import attention


# @a:mha
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """Split Q, K, V into `n_heads` column blocks, attend in each, concatenate, mix with W_O."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    size = Q.shape[1] // n_heads
    heads = [attention(Q[:, s], K[:, s], V[:, s], mask)
             for s in (slice(h * size, (h + 1) * size) for h in range(n_heads))]
    outputs, weights = zip(*heads)
    return np.concatenate(outputs, axis=1) @ W_O, list(weights)


# @a:demo
if __name__ == "__main__":
    # @a:shapes
    rng = np.random.default_rng(0)
    T, d, h = 6, 16, 4
    X = rng.normal(size=(T, d))
    W_Q, W_K, W_V, W_O = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
    out, weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, h, causal_mask(T))
    print(f"{h} heads of size {d // h}: output {out.shape}, {len(weights)} attention patterns")
    assert out.shape == X.shape  # same shape in and out, so blocks can be stacked

    # @a:one-head
    single, _ = attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask(T))
    one, _ = multi_head_attention(X, W_Q, W_K, W_V, W_O, 1, causal_mask(T))
    assert np.allclose(one, single @ W_O)  # one head is plain attention

    # @a:params
    params = sum(W.size for W in (W_Q, W_K, W_V, W_O))
    assert params == 4 * d * d  # 1 head or 16, the parameter count is the same
    print(f"{params} parameters, whatever the number of heads")

    # @a:specialists
    P = np.eye(T)  # token t is just "I am at position t"
    look_back = np.eye(T, k=1)  # key of position s looks like position s + 1
    W_Q2 = np.hstack([10 * P, 10 * P])  # both heads ask "who matches my position?"
    W_K2 = np.hstack([look_back, P])  # head 0 answers with the next position, head 1 with its own
    W_V2 = np.hstack([P, P])
    _, (prev, self_) = multi_head_attention(P, W_Q2, W_K2, W_V2, np.eye(2 * T), 2, causal_mask(T))
    print("head 0:", [f"{t}->{prev[t].argmax()}" for t in range(1, T)],
          "| head 1:", [f"{t}->{self_[t].argmax()}" for t in range(T)])
    assert np.array_equal(prev.argmax(axis=1)[1:], np.arange(T - 1))  # the previous token
    assert np.array_equal(self_.argmax(axis=1), np.arange(T))  # itself
