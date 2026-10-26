"""Embeddings in NumPy: a lookup table of vectors, the same as a one-hot matrix product."""

import numpy as np

# @a:reuse
from tokenization_numpy import build_vocab, encode
from vectors_numpy import dot, norm


# @a:one-hot
def one_hot(ids, size):
    """Token i → a row of `size` zeros with a 1 in column i."""
    return np.eye(size)[ids]


# @a:embed
def embed(ids, E):
    """Token i → row i of the embedding table E (V × d)."""
    return E[ids]


# @a:cosine
def cosine(u, v):
    """cos θ between two vectors: 1 = same direction, 0 = unrelated, −1 = opposite."""
    return dot(u, v) / (norm(u) * norm(v))


# @a:demo
if __name__ == "__main__":
    # @a:lookup
    rng = np.random.default_rng(0)
    vocab = build_vocab("abcdefghijklmnopqrstuvwxyz ")
    ids = encode("the cat sat", vocab)
    E = rng.normal(size=(len(vocab), 4))  # 27 tokens, 4 numbers each
    X = embed(ids, E)
    print(f"{len(ids)} tokens -> X of shape {X.shape}")
    assert np.allclose(X, one_hot(ids, len(vocab)) @ E)  # a lookup is a one-hot matmul
    assert np.array_equal(X[0], X[6])  # both t's get the same vector: no context yet

    # @a:gradient
    dX = rng.normal(size=X.shape)  # pretend gradient of the loss w.r.t. each looked-up row
    dE = np.zeros_like(E)
    np.add.at(dE, ids, dX)  # each row collects the gradients of its occurrences
    assert np.allclose(dE, one_hot(ids, len(vocab)).T @ dX)  # same as backprop through the matmul
    unused = sorted(set(range(len(vocab))) - set(ids.tolist()))
    assert np.all(dE[unused] == 0)  # tokens not in the batch don't move
    print(f"{len(unused)} of {len(vocab)} rows get no update from this batch")

    # @a:meaning
    words = ["king", "queen", "man", "woman", "apple"]
    W = np.array([  # hand-made features: royalty, maleness, edibility
        [0.9, 0.8, 0.0], [0.9, -0.8, 0.0], [0.1, 0.8, 0.0], [0.1, -0.8, 0.0], [0.0, 0.0, 0.9],
    ])
    target = W[0] - W[2] + W[3]  # king − man + woman
    scores = [cosine(target, w) for w in W]
    print({w: round(float(s), 2) for w, s in zip(words, scores)})
    assert words[int(np.argmax(scores))] == "queen"
    assert cosine(W[0], W[1]) > cosine(W[0], W[4])  # king is closer to queen than to apple
