"""k-fold cross-validation in NumPy: choose a setting without ever touching the test set."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse

# @a:data
rng = np.random.default_rng(0)
n, p = 400, 50
X = rng.normal(size=(n, p))
y = X[:, :5] @ np.array([3.0, -2.0, 1.5, 1.0, -1.0]) + rng.normal(scale=2.0, size=n)
X_train, y_train = X[:60], y[:60]  # 60 rows for 50 features: the penalty matters
X_test, y_test = X[60:], y[60:]  # a large test set, so the final check isn't luck


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


# @a:folds
k = 5
folds = np.array_split(rng.permutation(len(y_train)), k)

# @a:cv
alphas = np.array([1e-6, 0.1, 1, 10, 30, 100, 1000])
cv_mse = []
for alpha in alphas:
    scores = []
    for fold in folds:
        fit_rows = np.setdiff1d(np.arange(len(y_train)), fold)  # the other k − 1 folds
        w = ridge(X_train[fit_rows], y_train[fit_rows], alpha)
        scores.append(mse(y_train[fold], X_train[fold] @ w))
    cv_mse.append(np.mean(scores))
    print(f"α = {alpha:<6g}  CV MSE {cv_mse[-1]:7.2f}")

# @a:final
best = alphas[np.argmin(cv_mse)]
test_best = mse(y_test, X_test @ ridge(X_train, y_train, best))  # refit on all training rows
test_none = mse(y_test, X_test @ ridge(X_train, y_train, 1e-6))
print(f"chosen α = {best:g}: test MSE {test_best:.2f} (vs {test_none:.2f} with no penalty)")

# @a:checks
assert sorted(np.concatenate(folds)) == list(range(len(y_train)))  # each row validated once
assert 1e-6 < best < 1000  # CV found a penalty between the extremes
assert test_best < test_none  # and it carries over to data CV never saw
