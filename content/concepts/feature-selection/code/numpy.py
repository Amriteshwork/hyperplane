"""Feature selection with the Lasso in NumPy: let cross-validation decide which features stay."""

import numpy as np

# @a:reuse
from l1_regularization_numpy import lasso
from mean_squared_error_numpy import mse
from normal_equation_numpy import least_squares

# @a:data
rng = np.random.default_rng(0)
n, p, useful = 400, 30, [0, 1, 2, 3, 4]
X = rng.normal(size=(n, p))
y = 5 + X[:, useful] @ np.array([3.0, -2.5, 2.0, -1.5, 1.0]) + rng.normal(scale=2.0, size=n)
train, test = np.arange(60), np.arange(60, n)  # 60 rows for 30 features: easy to overfit


# @a:fit
def fit(rows, alpha):
    x_mean, y_mean = X[rows].mean(axis=0), y[rows].mean()
    w = lasso(X[rows] - x_mean, y[rows] - y_mean, alpha)
    return w, y_mean - x_mean @ w  # weights, intercept


# @a:cv
folds = np.array_split(rng.permutation(train), 5)
alphas = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
cv_mse = []
for alpha in alphas:
    scores = []
    for fold in folds:
        w, b = fit(np.setdiff1d(train, fold), alpha)  # train on the other four folds
        scores.append(mse(y[fold], X[fold] @ w + b))
    cv_mse.append(np.mean(scores))
best = alphas[np.argmin(cv_mse)]

# @a:select
w, b = fit(train, best)
selected = np.flatnonzero(w)
print(f"α = {best}: kept {len(selected)} of {p} features: {selected.tolist()}")

# @a:relaxed
ones = np.ones((len(train), 1))
refit = least_squares(np.hstack([ones, X[train][:, selected]]), y[train])  # OLS, kept only
all_ols = least_squares(np.hstack([ones, X[train]]), y[train])  # OLS, every feature
lasso_mse = mse(y[test], X[test] @ w + b)
relaxed_mse = mse(y[test], refit[0] + X[test][:, selected] @ refit[1:])
all_mse = mse(y[test], all_ols[0] + X[test] @ all_ols[1:])
print(f"test MSE: lasso {lasso_mse:.2f}, relaxed {relaxed_mse:.2f}, OLS on all {all_mse:.2f}")

# @a:checks
assert set(useful) <= set(selected)  # every useful feature survived...
assert len(selected) < p / 2  # ...and most useless ones were dropped
assert max(lasso_mse, relaxed_mse) < all_mse  # fewer features generalise better here
