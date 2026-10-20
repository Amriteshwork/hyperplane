"""k-fold cross-validation in PyTorch: choose a setting without ever touching the test set."""

import torch

# @a:reuse
from mean_squared_error_pytorch import mse

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n, p = 400, 50
X = torch.randn(n, p, dtype=f64)
true_w = torch.tensor([3.0, -2.0, 1.5, 1.0, -1.0], dtype=f64)
y = X[:, :5] @ true_w + 2.0 * torch.randn(n, dtype=f64)
X_train, y_train = X[:60], y[:60]  # 60 rows for 50 features: the penalty matters
X_test, y_test = X[60:], y[60:]  # a large test set, so the final check isn't luck


def ridge(X, y, alpha):
    return torch.linalg.solve(X.T @ X + alpha * torch.eye(X.shape[1], dtype=f64), X.T @ y)


# @a:folds
k = 5
folds = torch.tensor_split(torch.randperm(len(y_train)), k)

# @a:cv
alphas = [1e-6, 0.1, 1, 10, 30, 100, 1000]
cv_mse = []
for alpha in alphas:
    scores = []
    for fold in folds:
        keep = torch.ones(len(y_train), dtype=torch.bool)
        keep[fold] = False  # the other k − 1 folds
        w = ridge(X_train[keep], y_train[keep], alpha)
        scores.append(mse(y_train[fold], X_train[fold] @ w).item())
    cv_mse.append(sum(scores) / k)
    print(f"α = {alpha:<6g}  CV MSE {cv_mse[-1]:7.2f}")

# @a:final
best = alphas[min(range(len(alphas)), key=cv_mse.__getitem__)]
test_best = mse(y_test, X_test @ ridge(X_train, y_train, best)).item()  # refit on all rows
test_none = mse(y_test, X_test @ ridge(X_train, y_train, 1e-6)).item()
print(f"chosen α = {best:g}: test MSE {test_best:.2f} (vs {test_none:.2f} with no penalty)")

# @a:checks
assert sorted(torch.cat(folds).tolist()) == list(range(len(y_train)))  # each row validated once
assert 1e-6 < best < 1000  # CV found a penalty between the extremes
assert test_best < test_none  # and it carries over to data CV never saw
