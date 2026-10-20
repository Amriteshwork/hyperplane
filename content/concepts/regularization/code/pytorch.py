"""Regularisation in PyTorch: penalise large weights, and a too-flexible model stops overfitting."""

import torch

# @a:data
torch.manual_seed(0)
f64 = torch.float64
x = torch.cat([torch.linspace(-1, 1, 13, dtype=f64), torch.rand(200, dtype=f64) * 2 - 1])
y = torch.sin(3 * x) + 0.2 * torch.randn(len(x), dtype=f64)
train, test = torch.arange(13), torch.arange(13, len(x))
X = torch.linalg.vander(x, N=13)[:, 1:]  # x … x¹² + intercept: 13 weights, 13 rows
X = (X - X[train].mean(dim=0)) / X[train].std(dim=0, correction=0)  # one penalty fits all


# @a:fit
def fit(lam):
    Xc, yc = X[train] - X[train].mean(dim=0), y[train] - y[train].mean()
    w = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], dtype=f64), Xc.T @ yc)
    b = y[train].mean() - X[train].mean(dim=0) @ w  # the intercept is not penalised
    test_mse = ((X[test] @ w + b - y[test]) ** 2).mean().item()
    return test_mse, torch.linalg.norm(w).item()


# @a:sweep
results = {lam: fit(lam) for lam in (1e-8, 0.01, 0.1, 1.0, 100.0)}
for lam, (test_mse, size) in results.items():
    print(f"λ = {lam:<6g}  test MSE {test_mse:8.3f}   ‖w‖ {size:10.2f}")

# @a:checks
sizes = [size for _, size in results.values()]
assert sizes == sorted(sizes, reverse=True)  # a bigger penalty always means smaller weights
best = min(results, key=lambda lam: results[lam][0])
assert 1e-8 < best < 100.0  # neither no penalty nor a crushing one generalises best
assert results[best][0] < results[1e-8][0] / 2  # the penalty tames the overfit model
