"""Feature selection with the Lasso in PyTorch: let cross-validation decide which features stay."""

import torch

# @a:reuse
from l1_regularization_pytorch import lasso
from mean_squared_error_pytorch import mse
from normal_equation_pytorch import least_squares

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n, p, useful = 400, 30, [0, 1, 2, 3, 4]
X = torch.randn(n, p, dtype=f64)
true_w = torch.tensor([3.0, -2.5, 2.0, -1.5, 1.0], dtype=f64)
y = 5 + X[:, useful] @ true_w + 2.0 * torch.randn(n, dtype=f64)
train, test = torch.arange(60), torch.arange(60, n)  # 60 rows for 30 features: easy to overfit


# @a:fit
def fit(rows, alpha):
    x_mean, y_mean = X[rows].mean(dim=0), y[rows].mean()
    w = lasso(X[rows] - x_mean, y[rows] - y_mean, alpha)
    return w, y_mean - x_mean @ w  # weights, intercept


# @a:cv
folds = torch.tensor_split(train[torch.randperm(len(train))], 5)
alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
cv_mse = []
for alpha in alphas:
    scores = []
    for fold in folds:
        rest = train[~torch.isin(train, fold)]  # train on the other four folds
        w, b = fit(rest, alpha)
        scores.append(float(mse(y[fold], X[fold] @ w + b)))
    cv_mse.append(sum(scores) / len(scores))
best = alphas[min(range(len(alphas)), key=cv_mse.__getitem__)]

# @a:select
w, b = fit(train, best)
selected = torch.nonzero(w).flatten()
print(f"α = {best}: kept {len(selected)} of {p} features: {selected.tolist()}")

# @a:relaxed
ones = torch.ones(len(train), 1, dtype=f64)
refit = least_squares(torch.hstack([ones, X[train][:, selected]]), y[train])  # OLS, kept only
all_ols = least_squares(torch.hstack([ones, X[train]]), y[train])  # OLS, every feature
lasso_mse = float(mse(y[test], X[test] @ w + b))
relaxed_mse = float(mse(y[test], refit[0] + X[test][:, selected] @ refit[1:]))
all_mse = float(mse(y[test], all_ols[0] + X[test] @ all_ols[1:]))
print(f"test MSE: lasso {lasso_mse:.2f}, relaxed {relaxed_mse:.2f}, OLS on all {all_mse:.2f}")

# @a:checks
assert set(useful) <= set(selected.tolist())  # every useful feature survived...
assert len(selected) < p / 2  # ...and most useless ones were dropped
assert max(lasso_mse, relaxed_mse) < all_mse  # fewer features generalise better here
