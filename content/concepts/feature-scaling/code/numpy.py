"""Feature scaling in NumPy: why a penalised model needs features on the same scale."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 2))
y = 2 * X[:, 0] + 2 * X[:, 1] + rng.normal(scale=0.5, size=n)  # equally important features
X[:, 1] /= 1000  # same information, different units (e.g. metres → kilometres)
train, test = np.arange(300), np.arange(300, n)


# @a:ridge
def ridge_test_mse(X, alpha=10.0):
    Xc, yc = X[train] - X[train].mean(axis=0), y[train] - y[train].mean()
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(2), Xc.T @ yc)
    b = y[train].mean() - X[train].mean(axis=0) @ w
    return np.mean((X[test] @ w + b - y[test]) ** 2), w


# @a:standardise
mean, std = X[train].mean(axis=0), X[train].std(axis=0)  # learned from training rows only
X_scaled = (X - mean) / std

# @a:compare
raw_mse, raw_w = ridge_test_mse(X)
scaled_mse, scaled_w = ridge_test_mse(X_scaled)
print(f"unscaled: test MSE {raw_mse:.3f}, weights {raw_w.round(3)}")
print(f"scaled:   test MSE {scaled_mse:.3f}, weights {scaled_w.round(3)}")

# @a:checks
assert abs(raw_w[1]) * std[1] < 0.1  # the km feature needs a huge weight: crushed
assert raw_mse > 5 * scaled_mse  # ...and lost half the signal
assert abs(scaled_w[0] - scaled_w[1]) < 0.2  # scaled: both treated alike, as they should be
