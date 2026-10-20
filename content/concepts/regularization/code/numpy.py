"""Regularisation in NumPy: penalise large weights, and a too-flexible model stops overfitting."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
x = np.concatenate([np.linspace(-1, 1, 13), rng.uniform(-1, 1, size=200)])
y = np.sin(3 * x) + rng.normal(scale=0.2, size=len(x))
train, test = np.arange(13), np.arange(13, len(x))
X = np.vander(x, 13, increasing=True)[:, 1:]  # x … x¹² + intercept: 13 weights, 13 rows
X = (X - X[train].mean(axis=0)) / X[train].std(axis=0)  # scale, so one penalty fits all


# @a:fit
def fit(lam):
    Xc, yc = X[train] - X[train].mean(axis=0), y[train] - y[train].mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)  # loss + λ‖w‖²
    b = y[train].mean() - X[train].mean(axis=0) @ w  # the intercept is not penalised
    test_mse = np.mean((X[test] @ w + b - y[test]) ** 2)
    return test_mse, np.linalg.norm(w)


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
