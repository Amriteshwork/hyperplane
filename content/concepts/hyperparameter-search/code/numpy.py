"""Hyperparameter search in NumPy: tune two interacting settings by cross-validation."""

import numpy as np

# @a:reuse
from combined_penalties_numpy import elastic_net
from mean_squared_error_numpy import mse

# @a:data
rng = np.random.default_rng(0)
n, p = 400, 20
signal = rng.normal(size=(n, 2))
X = np.hstack([np.repeat(signal, 3, axis=1) + 0.3 * rng.normal(size=(n, 6)),  # 2 groups of 3
               rng.normal(size=(n, p - 6))])  # 14 useless features
y = signal @ np.array([3.0, -2.0]) + rng.normal(scale=2.0, size=n)
train, test = np.arange(80), np.arange(80, n)
folds = np.array_split(rng.permutation(train), 5)


# @a:cv
def cv_mse(alpha, l1_ratio):
    """Mean validation MSE over the 5 folds of the training rows."""
    scores = []
    for fold in folds:
        rows = np.setdiff1d(train, fold)  # the other four folds
        x_mean, y_mean = X[rows].mean(axis=0), y[rows].mean()
        w = elastic_net(X[rows] - x_mean, y[rows] - y_mean, alpha, l1_ratio, tol=1e-4)
        scores.append(mse(y[fold], (X[fold] - x_mean) @ w + y_mean))
    return np.mean(scores)


# @a:grid
alphas, ratios = np.logspace(-1.5, 1, 6), [0.1, 0.5, 1.0]
grid = {(a, r): cv_mse(a, r) for a in alphas for r in ratios}  # 18 settings × 5 folds
best = min(grid, key=grid.get)
print(f"grid:   best α = {best[0]:.3f}, l1_ratio = {best[1]}, CV MSE {grid[best]:.2f}")

# @a:random
trials = [(10 ** rng.uniform(-1.5, 1), rng.uniform(0.1, 1.0)) for _ in range(6)]
sampled = {t: cv_mse(*t) for t in trials}  # about a third of the grid's budget
lucky = min(sampled, key=sampled.get)
print(f"random: best α = {lucky[0]:.3f}, l1_ratio = {lucky[1]:.2f}, CV MSE {sampled[lucky]:.2f}")


# @a:final
def test_mse(alpha, l1_ratio):
    x_mean, y_mean = X[train].mean(axis=0), y[train].mean()
    w = elastic_net(X[train] - x_mean, y[train] - y_mean, alpha, l1_ratio, tol=1e-6)  # all rows
    return mse(y[test], (X[test] - x_mean) @ w + y_mean)


worst = max(grid, key=grid.get)
print(f"test MSE: best {test_mse(*best):.2f}, worst grid setting {test_mse(*worst):.2f}")

# @a:checks
assert test_mse(*best) < test_mse(*worst)  # what CV preferred also wins on unseen rows
assert test_mse(*best) < test_mse(0.03, 1.0)  # ...and beats a barely-penalised model
assert sampled[lucky] < 1.1 * grid[best]  # a third of the budget gets close
