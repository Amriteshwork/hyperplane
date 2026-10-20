"""Hyperparameter search in PyTorch: tune two interacting settings by cross-validation."""

import torch

# @a:reuse
from combined_penalties_pytorch import elastic_net
from mean_squared_error_pytorch import mse

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n, p = 400, 20
signal = torch.randn(n, 2, dtype=f64)
groups = signal.repeat_interleave(3, dim=1) + 0.3 * torch.randn(n, 6, dtype=f64)  # 2 groups of 3
X = torch.hstack([groups, torch.randn(n, p - 6, dtype=f64)])  # ...plus 14 useless features
y = signal @ torch.tensor([3.0, -2.0], dtype=f64) + 2.0 * torch.randn(n, dtype=f64)
train, test = torch.arange(80), torch.arange(80, n)
folds = torch.tensor_split(train[torch.randperm(len(train))], 5)


# @a:cv
def cv_mse(alpha, l1_ratio):
    """Mean validation MSE over the 5 folds of the training rows."""
    scores = []
    for fold in folds:
        rows = train[~torch.isin(train, fold)]  # the other four folds
        x_mean, y_mean = X[rows].mean(dim=0), y[rows].mean()
        w = elastic_net(X[rows] - x_mean, y[rows] - y_mean, alpha, l1_ratio, tol=1e-4)
        scores.append(mse(y[fold], (X[fold] - x_mean) @ w + y_mean).item())
    return sum(scores) / len(scores)


# @a:grid
alphas, ratios = torch.logspace(-1.5, 1, 6).tolist(), [0.1, 0.5, 1.0]
grid = {(a, r): cv_mse(a, r) for a in alphas for r in ratios}  # 18 settings × 5 folds
best = min(grid, key=grid.get)
print(f"grid:   best α = {best[0]:.3f}, l1_ratio = {best[1]}, CV MSE {grid[best]:.2f}")

# @a:random
draws = torch.rand(6, 2, dtype=f64)
trials = [(10 ** (2.5 * a.item() - 1.5), 0.1 + 0.9 * r.item()) for a, r in draws]
sampled = {t: cv_mse(*t) for t in trials}  # about a third of the grid's budget
lucky = min(sampled, key=sampled.get)
print(f"random: best α = {lucky[0]:.3f}, l1_ratio = {lucky[1]:.2f}, CV MSE {sampled[lucky]:.2f}")


# @a:final
def test_mse(alpha, l1_ratio):
    x_mean, y_mean = X[train].mean(dim=0), y[train].mean()
    w = elastic_net(X[train] - x_mean, y[train] - y_mean, alpha, l1_ratio, tol=1e-6)  # all rows
    return mse(y[test], (X[test] - x_mean) @ w + y_mean).item()


worst = max(grid, key=grid.get)
print(f"test MSE: best {test_mse(*best):.2f}, worst grid setting {test_mse(*worst):.2f}")

# @a:checks
assert test_mse(*best) < test_mse(*worst)  # what CV preferred also wins on unseen rows
assert test_mse(*best) < test_mse(0.03, 1.0)  # ...and beats a barely-penalised model
assert sampled[lucky] < 1.1 * grid[best]  # a third of the budget gets close
