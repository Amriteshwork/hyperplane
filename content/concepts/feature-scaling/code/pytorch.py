"""Feature scaling in PyTorch: why a penalised model needs features on the same scale."""

import torch

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n = 400
X = torch.randn(n, 2, dtype=f64)
y = 2 * X[:, 0] + 2 * X[:, 1] + 0.5 * torch.randn(n, dtype=f64)  # equally important features
X[:, 1] /= 1000  # same information, different units (e.g. metres → kilometres)
train, test = torch.arange(300), torch.arange(300, n)


# @a:ridge
def ridge_test_mse(X, alpha=10.0):
    Xc, yc = X[train] - X[train].mean(dim=0), y[train] - y[train].mean()
    w = torch.linalg.solve(Xc.T @ Xc + alpha * torch.eye(2, dtype=f64), Xc.T @ yc)
    b = y[train].mean() - X[train].mean(dim=0) @ w
    return ((X[test] @ w + b - y[test]) ** 2).mean().item(), w


# @a:standardise
mean, std = X[train].mean(dim=0), X[train].std(dim=0, correction=0)  # training rows only
X_scaled = (X - mean) / std

# @a:compare
raw_mse, raw_w = ridge_test_mse(X)
scaled_mse, scaled_w = ridge_test_mse(X_scaled)
print(f"unscaled: test MSE {raw_mse:.3f}, weights {raw_w.round(decimals=3).tolist()}")
print(f"scaled:   test MSE {scaled_mse:.3f}, weights {scaled_w.round(decimals=3).tolist()}")

# @a:checks
assert abs(raw_w[1]) * std[1] < 0.1  # the km feature needs a huge weight: crushed
assert raw_mse > 5 * scaled_mse  # ...and lost half the signal
assert abs(scaled_w[0] - scaled_w[1]) < 0.2  # scaled: both treated alike, as they should be
