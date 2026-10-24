"""Stochastic gradient descent in NumPy: step on one row (or a small batch) at a time."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse


# @a:sgd-epoch
def sgd_epoch(X, y, w, b, learning_rate, rng, batch_size=1):
    """One pass over the shuffled rows, one MSE gradient step per batch. Returns the new w, b."""
    order = rng.permutation(len(y))  # a fresh random order every pass
    for start in range(0, len(y), batch_size):
        rows = order[start:start + batch_size]
        error = X[rows] @ w + b - y[rows]
        w = w - learning_rate * 2 * X[rows].T @ error / len(rows)  # this batch's estimate of ∇MSE
        b = b - learning_rate * 2 * error.mean()
    return w, b


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 10_000
    X = rng.normal(size=(n, 2))
    y = 4.0 + X @ np.array([2.0, -3.0]) + rng.normal(scale=0.5, size=n)

    # @a:unbiased
    w = np.zeros(2)
    full = 2 * X.T @ (X @ w - y) / n  # the true gradient: every row
    rows = rng.integers(n, size=5000)
    one_row = 2 * X[rows] * (X[rows] @ w - y[rows])[:, None]  # 5000 single-row estimates
    print("full gradient:", full.round(2), "| one row, e.g.:", one_row[0].round(2),
          "| average of one-row estimates:", one_row.mean(axis=0).round(2))
    assert np.allclose(one_row.mean(axis=0), full, atol=0.5)  # noisy, but right on average

    # @a:batches
    results = {}
    for batch_size in (1, 32, n):
        w, b = sgd_epoch(X, y, np.zeros(2), 0.0, learning_rate=0.01, rng=rng, batch_size=batch_size)
        results[batch_size] = mse(y, X @ w + b)
        print(f"batch {batch_size:>5}: {-(-n // batch_size):>5} steps in one pass, MSE {results[batch_size]:.4f}")
    assert results[1] < 0.3 and results[32] < 0.3  # many cheap steps: already at the noise floor
    assert results[n] > 10  # one careful step: barely started
