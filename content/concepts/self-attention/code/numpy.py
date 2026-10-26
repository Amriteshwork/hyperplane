"""Self-attention in NumPy: softmax(QKᵀ/√d)·V, checked against a plain loop."""

import numpy as np

# @a:reuse
from softmax_numpy import softmax


# @a:attention
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention; returns the outputs and the weights, so we can look at them."""
    scores = Q @ K.T / np.sqrt(K.shape[1])  # (T, T): every query against every key
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # masked pairs get weight exactly 0
    weights = softmax(scores)  # each row sums to 1
    return weights @ V, weights


# @a:demo
if __name__ == "__main__":
    # @a:lookup
    keys = np.eye(3) * 4  # three very different keys
    values = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    queries = np.array([keys[1], np.zeros(3)])  # a sharp question, then a vague one
    out, w = attention(queries, keys, values)
    print("weights", w.round(3).tolist(), "-> outputs", out.round(2).tolist())
    assert np.allclose(out[0], values[1], atol=0.01)  # matches key 1: gets value 1
    assert np.allclose(out[1], values.mean(axis=0))  # matches nothing: gets the average

    # @a:projections
    rng = np.random.default_rng(0)
    T, d, d_k = 5, 8, 4
    X = rng.normal(size=(T, d))  # five token embeddings
    W_Q, W_K, W_V = (rng.normal(size=(d, d_k)) / np.sqrt(d) for _ in range(3))
    out, w = attention(X @ W_Q, X @ W_K, X @ W_V)
    assert out.shape == (T, d_k) and np.allclose(w.sum(axis=1), 1)

    # @a:loop
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    for t in range(T):  # one token at a time: score, softmax, blend
        s = np.array([Q[t] @ K[j] for j in range(T)]) / np.sqrt(d_k)
        a = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
        assert np.allclose(out[t], sum(a[j] * V[j] for j in range(T)))

    # @a:scaling
    for dim in (4, 64, 1024):
        q, k = rng.normal(size=(2, 10_000, dim))
        dots = (q * k).sum(axis=1)  # 10,000 random query·key scores
        print(f"d={dim:4d}: score std {dots.std():5.1f}, after /√d {dots.std() / np.sqrt(dim):.2f}")
        assert abs(dots.std() / np.sqrt(dim) - 1) < 0.05  # √d brings the spread back to 1
    raw = softmax(rng.normal(size=(1, 8)) * np.sqrt(1024))  # unscaled scores at d=1024
    print(f"unscaled softmax at d=1024 puts {raw.max():.3f} of the weight on one token")
    assert raw.max() > 0.99  # practically one-hot: gradients through it vanish

    # @a:order
    perm = rng.permutation(T)
    shuffled, _ = attention(X[perm] @ W_Q, X[perm] @ W_K, X[perm] @ W_V)
    assert np.allclose(shuffled, out[perm])  # same outputs, just shuffled: order is invisible
    print("shuffling the tokens only shuffles the outputs")
