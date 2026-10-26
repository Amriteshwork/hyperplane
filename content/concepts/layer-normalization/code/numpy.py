"""Layer normalization in NumPy: standardise every token's vector on its own, then rescale."""

import numpy as np

# @a:reuse
from residual_connections_numpy import residual


# @a:layer-norm
def layer_norm(X, gamma, beta, eps=1e-5):
    """Each row (token) to mean 0 and variance 1 across its features, then × γ + β per feature."""
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta


# @a:demo
if __name__ == "__main__":
    # @a:rows
    rng = np.random.default_rng(0)
    d = 8
    X = rng.normal(size=(4, d)) * [[1], [10], [0.1], [1000]] + [[0], [5], [-3], [200]]
    ones, zeros = np.ones(d), np.zeros(d)
    Y = layer_norm(X, ones, zeros)
    print("row means", Y.mean(axis=1).round(6), "row stds", Y.std(axis=1).round(3))
    assert np.allclose(Y.mean(axis=1), 0) and np.allclose(Y.std(axis=1), 1, atol=1e-2)
    assert np.all(layer_norm(np.full((1, d), 4.2), ones, zeros) == 0)  # constant row: ε avoids 0/0

    # @a:invariance
    x = X[:1]
    assert np.allclose(layer_norm(7 * x + 3, ones, zeros), layer_norm(x, ones, zeros), atol=1e-4)

    # @a:not-columns
    others = X.copy()
    others[1:] = rng.normal(size=(3, d))  # replace every *other* token
    assert np.allclose(layer_norm(others, ones, zeros)[0], Y[0])  # row 0 doesn't care
    columns = (X - X.mean(axis=0)) / X.std(axis=0)  # feature scaling standardises columns...
    assert not np.allclose(((others - others.mean(axis=0)) / others.std(axis=0))[0], columns[0])

    # @a:stream
    T = 16
    h = rng.normal(size=(T, 64))
    for depth in range(1, 49):
        W = rng.normal(size=(64, 64)) / np.sqrt(64)
        h = residual(h, lambda z: layer_norm(z, np.ones(64), np.zeros(64)) @ W)  # pre-LN branch
        if depth in (1, 12, 48):
            print(f"after {depth:2d} blocks: residual stream size {np.linalg.norm(h, axis=1).mean():5.1f},"
                  f" what the next branch reads {np.linalg.norm(layer_norm(h, 1, 0), axis=1).mean():.1f}")
    assert np.linalg.norm(h, axis=1).mean() > 40  # the stream keeps growing with depth...
    assert np.allclose(np.linalg.norm(layer_norm(h, 1, 0), axis=1), 8, atol=1e-3)  # ...LN: √64 = 8
