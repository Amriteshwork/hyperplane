"""Causal masking in NumPy: a lower-triangular mask so no token can see the future."""

import numpy as np

# @a:reuse
from self_attention_numpy import attention


# @a:mask
def causal_mask(T):
    """True on and below the diagonal: token t may look at tokens 0..t."""
    return np.tril(np.ones((T, T), dtype=bool))


# @a:demo
if __name__ == "__main__":
    # @a:weights
    rng = np.random.default_rng(0)
    T, d = 5, 4
    X = rng.normal(size=(T, d))
    W_Q, W_K, W_V = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    out, w = attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask(T))
    print(w.round(2))
    assert np.all(w[~causal_mask(T)] == 0)  # nothing above the diagonal
    assert np.allclose(w.sum(axis=1), 1) and w[0, 0] == 1  # token 0 can only see itself

    # @a:no-peeking
    future = X.copy()
    future[3:] = rng.normal(size=(T - 3, d))  # rewrite tokens 3 and 4
    out2, _ = attention(future @ W_Q, future @ W_K, future @ W_V, causal_mask(T))
    assert np.allclose(out2[:3], out[:3])  # tokens 0–2 can't tell anything changed
    assert not np.allclose(out2[3:], out[3:])
    print("changing the future leaves the past untouched")

    # @a:running-mean
    V = X @ W_V
    avg, _ = attention(np.zeros((T, d)), X @ W_K, V, causal_mask(T))  # all scores equal
    running_mean = np.cumsum(V, axis=0) / np.arange(1, T + 1)[:, None]
    assert np.allclose(avg, running_mean)  # each position averages everything so far
