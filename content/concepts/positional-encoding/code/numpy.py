"""Positional encoding in NumPy: sine and cosine clocks that tell attention where each token is."""

import numpy as np

# @a:reuse
from self_attention_numpy import attention


# @a:sinusoidal
def sinusoidal(T, d):
    """PE[t, 2i] = sin(t·ωᵢ), PE[t, 2i+1] = cos(t·ωᵢ), with ωᵢ = 1 / 10000^(2i/d)."""
    t = np.arange(T)[:, None]
    freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)  # fast clocks first, slow clocks last
    PE = np.zeros((T, d))
    PE[:, 0::2], PE[:, 1::2] = np.sin(t * freq), np.cos(t * freq)
    return PE


# @a:demo
if __name__ == "__main__":
    # @a:word-order
    rng = np.random.default_rng(0)
    d = 16
    E = dict(zip(["dog", "bites", "man"], rng.normal(size=(3, d))))
    W_Q, W_K, W_V = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))

    def read(words, pe):
        """Self-attention over the sentence; returns the output vector for 'dog'."""
        X = np.array([E[w] for w in words]) + pe
        return attention(X @ W_Q, X @ W_K, X @ W_V)[0][words.index("dog")]

    no_pe = np.zeros((3, d))
    assert np.allclose(read(["dog", "bites", "man"], no_pe), read(["man", "bites", "dog"], no_pe))
    pe = sinusoidal(3, d)
    assert not np.allclose(read(["dog", "bites", "man"], pe), read(["man", "bites", "dog"], pe))
    print("without positions, the biter and the bitten look the same; with them, they don't")

    # @a:unique
    PE = sinusoidal(100, d)
    gaps = np.linalg.norm(PE[:, None] - PE[None], axis=2) + np.eye(100)  # ignore self-distances
    assert gaps.min() > 0.1  # every position gets its own fingerprint
    assert np.abs(PE).max() <= 1  # and the values stay small, whatever the length

    # @a:relative
    k = 3
    dots = PE[:-k] @ PE[k:].T  # all pairwise similarities...
    assert np.allclose(np.diag(dots), dots[0, 0])  # ...but t vs t+3 is the same for every t
    w = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    R = np.zeros((d, d))  # one 2×2 rotation per clock, by angle k·ω
    for i, a in enumerate(k * w):
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    assert np.allclose(PE[:-k] @ R, PE[k:])  # moving 3 steps is the same rotation everywhere
    print("PE[t + 3] = PE[t] @ R for every t: a fixed rotation, so offsets are easy to learn")
