"""Sparsity in NumPy: as the L1 penalty falls, features join the model one at a time."""

import numpy as np

# @a:reuse
from l1_regularization_numpy import lasso

# @a:data
rng = np.random.default_rng(0)
n, p = 300, 6
X = rng.normal(size=(n, p))
y = X @ np.array([4.0, -3.0, 2.0, 1.0, 0.5, 0.0]) + rng.normal(size=n)
X, y = X - X.mean(axis=0), y - y.mean()  # centred: no intercept

# @a:alpha-max
alpha_max = np.max(np.abs(X.T @ y)) / n  # every weight is 0 from here up
print(f"alpha_max = {alpha_max:.3f}")

# @a:path
alphas = alpha_max * np.logspace(0, -3, 13)  # from all-zero down to almost no penalty
path = np.array([lasso(X, y, alpha) for alpha in alphas])
for alpha, w in zip(alphas, path):
    print(f"α = {alpha:7.4f}  non-zero: {np.count_nonzero(w)}  w = {w.round(2)}")

# @a:entry
first_used = [np.flatnonzero(path[:, j])[:1] for j in range(p)]  # first α where j is used
order = sorted(range(p), key=lambda j: first_used[j][0] if len(first_used[j]) else len(alphas))
print("features enter in order:", order)

# @a:ridge
ridge = np.linalg.solve(X.T @ X + 1e4 * np.eye(p), X.T @ y)  # a crushing L2 penalty

# @a:checks
counts = [np.count_nonzero(w) for w in path]
assert counts[0] == 0 and counts[-1] >= 5  # from nothing to (almost) everything
assert counts == sorted(counts)  # lowering α only ever added features here
assert order[:5] == [0, 1, 2, 3, 4]  # the biggest true effects enter first
assert np.all(ridge != 0)  # L2 shrinks weights to tiny values, but never to 0
